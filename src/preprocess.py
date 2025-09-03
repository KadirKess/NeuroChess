import os
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Local imports
from src.all_moves import get_all_legal_moves

_piece_to_channel = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,  # white
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,  # black
}


def fen_to_board_tensor_int8(fen: str):
    """
    Parse FEN string to a (18,8,8) np.int8 tensor.
    This is intentionally small and allocation-friendly.
    """
    # early guard
    if not isinstance(fen, str) or len(fen) == 0:
        return np.zeros((18, 8, 8), dtype=np.int8)

    parts = fen.split()
    board_part = parts[0]
    # side to move is parts[1] ('w' or 'b'), castling rights parts[2], ep parts[3]
    rows = board_part.split("/")
    tensor = np.zeros((18, 8, 8), dtype=np.int8)

    for r, row in enumerate(rows):
        c = 0
        for ch in row:
            if ch.isdigit():
                c += int(ch)
            else:
                channel = _piece_to_channel.get(ch)
                if channel is not None:
                    tensor[channel, r, c] = 1
                c += 1

    # side to move
    if len(parts) > 1 and parts[1] == "w":
        tensor[12, :, :] = 1
    # castling rights
    castling = parts[2] if len(parts) > 2 else "-"
    if "K" in castling:
        tensor[13, :, :] = 1
    if "Q" in castling:
        tensor[14, :, :] = 1
    if "k" in castling:
        tensor[15, :, :] = 1
    if "q" in castling:
        tensor[16, :, :] = 1
    # en-passant
    ep = parts[3] if len(parts) > 3 else "-"
    if ep != "-" and len(ep) >= 2:
        file_char = ord(ep[0]) - ord("a")
        rank_char = int(ep[1])  # '3' -> 3
        # compute row, col: rank '8' is row 0
        row_idx = 8 - rank_char
        col_idx = file_char
        if 0 <= row_idx < 8 and 0 <= col_idx < 8:
            tensor[17, row_idx, col_idx] = 1

    return tensor


# helper for Pool.map
def _process_single_row(args):
    fen, cp, mate, line, move_to_idx = args
    # board tensor
    board_t = fen_to_board_tensor_int8(fen)
    # game state + value
    if mate is None or mate == 0:
        game_state = 0
        value = (cp / 100.0) if (cp is not None and not np.isnan(cp)) else 0.0
    elif mate > 0:
        game_state = 1
        value = mate
    else:
        game_state = 2
        value = abs(mate)

    # first move
    first_move = ""
    if isinstance(line, str) and len(line) > 0:
        # take the first token in the moves string
        first_move = line.split()[0] if len(line.split()) > 0 else ""
    best_move_idx = move_to_idx.get(first_move, -1)

    return board_t, game_state, value, best_move_idx


def preprocess_data_in_chunks(
    parquet_path: str,
    destination_path: str,
    chunk_size: int = 100_000,
    num_workers: int | None = None,
    skip_unknown_moves: bool = True,
):
    """
    Faster preprocessing:
    - fast fen parser
    - multiprocessing per chunk
    - writes into memmap files incrementally
    - skip_unknown_moves: whether to exclude rows where first move not found in move map
    """
    os.makedirs(destination_path, exist_ok=True)

    parquet_file = pq.ParquetFile(parquet_path)
    num_rows = parquet_file.metadata.num_rows

    all_possible_moves = get_all_legal_moves()
    move_to_idx = {move: i for i, move in enumerate(all_possible_moves)}
    num_moves = len(all_possible_moves)
    print(f"Found {num_moves} possible moves. Preparing memmaps for {num_rows} rows...")

    # Memmaps (int8 for boards, int8 for game state, float32 for values, int64 for moves)
    boards_mm = np.memmap(
        f"{destination_path}/board_tensors.npy",
        dtype=np.int8,
        mode="w+",
        shape=(num_rows, 18, 8, 8),
    )
    states_mm = np.memmap(
        f"{destination_path}/game_state_target.npy",
        dtype=np.int8,
        mode="w+",
        shape=(num_rows,),
    )
    values_mm = np.memmap(
        f"{destination_path}/value_target.npy",
        dtype=np.float32,
        mode="w+",
        shape=(num_rows,),
    )
    moves_mm = np.memmap(
        f"{destination_path}/best_move.npy",
        dtype=np.int64,
        mode="w+",
        shape=(num_rows,),
    )

    if num_workers is None:
        num_workers = max(1, min(cpu_count() - 1, 8))

    start_idx = 0
    iterator = parquet_file.iter_batches(
        batch_size=chunk_size, columns=["fen", "cp", "mate", "line"]
    )
    total_batches = (num_rows + chunk_size - 1) // chunk_size
    print(f"Processing in {total_batches} chunks using {num_workers} workers...")

    for batch_idx, batch in enumerate(
        tqdm(iterator, total=total_batches, desc="Chunks")
    ):
        chunk_df = batch.to_pandas()
        n_chunk = len(chunk_df)
        args_iter = [
            (fen, cp, mate, line, move_to_idx)
            for fen, cp, mate, line in zip(
                chunk_df["fen"], chunk_df["cp"], chunk_df["mate"], chunk_df["line"]
            )
        ]

        # Parallel processing of rows
        with Pool(processes=num_workers) as pool:
            # use imap to stream results
            results = list(
                tqdm(
                    pool.imap(_process_single_row, args_iter),
                    total=n_chunk,
                    desc=f"Parsing chunk {batch_idx+1}/{total_batches}",
                )
            )

        # Unzip results
        boards_chunk = np.empty((n_chunk, 18, 8, 8), dtype=np.int8)
        states_chunk = np.empty((n_chunk,), dtype=np.int8)
        values_chunk = np.empty((n_chunk,), dtype=np.float32)
        moves_chunk = np.empty((n_chunk,), dtype=np.int64)

        for i, (b, s, v, m) in enumerate(results):
            boards_chunk[i] = b
            states_chunk[i] = s
            values_chunk[i] = v
            moves_chunk[i] = m

        # Optionally skip unknown moves
        if skip_unknown_moves:
            keep_mask = moves_chunk != -1
            kept = keep_mask.sum()
            if kept < len(moves_chunk):
                # write only kept rows
                end_idx = start_idx + kept
                boards_mm[start_idx:end_idx] = boards_chunk[keep_mask]
                states_mm[start_idx:end_idx] = states_chunk[keep_mask]
                values_mm[start_idx:end_idx] = values_chunk[keep_mask]
                moves_mm[start_idx:end_idx] = moves_chunk[keep_mask]
                start_idx = end_idx
            else:
                end_idx = start_idx + n_chunk
                boards_mm[start_idx:end_idx] = boards_chunk
                states_mm[start_idx:end_idx] = states_chunk
                values_mm[start_idx:end_idx] = values_chunk
                moves_mm[start_idx:end_idx] = moves_chunk
                start_idx = end_idx
        else:
            end_idx = start_idx + n_chunk
            boards_mm[start_idx:end_idx] = boards_chunk
            states_mm[start_idx:end_idx] = states_chunk
            values_mm[start_idx:end_idx] = values_chunk
            moves_mm[start_idx:end_idx] = moves_chunk
            start_idx = end_idx

        # flush per chunk (safe for interruptions)
        boards_mm.flush()
        states_mm.flush()
        values_mm.flush()
        moves_mm.flush()

        print(
            f"Chunk {batch_idx+1}/{total_batches} processed. Total written rows so far: {start_idx}"
        )

    # If we skipped unknown rows, start_idx might be < num_rows
    if start_idx < num_rows:
        # resize - create final arrays with the actual length
        actual = start_idx
        print(f"Shrinking arrays to actual size {actual} (removed unknown moves).")
        # create final files
        np.save(
            f"{destination_path}/board_tensors_final.npy",
            np.array(boards_mm[:actual], dtype=np.int8),
        )
        np.save(
            f"{destination_path}/game_state_target_final.npy",
            np.array(states_mm[:actual], dtype=np.int8),
        )
        np.save(
            f"{destination_path}/value_target_final.npy",
            np.array(values_mm[:actual], dtype=np.float32),
        )
        np.save(
            f"{destination_path}/best_move_final.npy",
            np.array(moves_mm[:actual], dtype=np.int64),
        )
        print(
            "Saved final (shrunk) numpy files. You can delete the intermediate memmap files if desired."
        )
    else:
        print(
            "Preprocessing complete. All rows written to memmaps (no unknown moves found or skip disabled)."
        )
