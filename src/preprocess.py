# Imports

import sys
import numpy as np
import chess
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import os

# Local imports

from src.all_moves import get_all_legal_moves


def _get_first_move_from_line(line: str) -> str:
    """
    Extract the first move from a line of moves.
    """
    moves = line.split()
    if moves:
        return moves[0]
    return ""


def _get_board_tensor(fen: str) -> np.ndarray:
    """Convert a FEN string to a board tensor (12, 8, 8)."""

    board = chess.Board(fen=fen)
    board_tensor = np.zeros((18, 8, 8), dtype=np.float32)

    # Channels 0-11 are for pieces
    piece_to_channel = {
        # White pieces: 0-5
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        # Black pieces: 6-11
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }

    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            row = i // 8
            col = i % 8
            channel = piece_to_channel[(piece.piece_type, piece.color)]
            board_tensor[channel, row, col] = 1.0

    # Channel 12 is for the side to move
    if board.turn == chess.WHITE:
        board_tensor[12, :, :] = 1.0

    # Channel 13 is for white king side castling
    if board.has_kingside_castling_rights(chess.WHITE):
        board_tensor[13, :, :] = 1.0

    # Channel 14 is for white queen side castling
    if board.has_queenside_castling_rights(chess.WHITE):
        board_tensor[14, :, :] = 1.0

    # Channel 15 is for black king side castling
    if board.has_kingside_castling_rights(chess.BLACK):
        board_tensor[15, :, :] = 1.0

    # Channel 16 is for black queen side castling
    if board.has_queenside_castling_rights(chess.BLACK):
        board_tensor[16, :, :] = 1.0

    # Channel 17 is for en passant
    if board.ep_square is not None:
        row, col = board.ep_square // 8, board.ep_square % 8
        board_tensor[17, row, col] = 1.0

    return board_tensor


def preprocess_data_in_chunks(
    parquet_path: str, destination_path: str, chunk_size: int = 100_000
):
    print("Initializing...")
    os.makedirs(destination_path, exist_ok=True)

    # Open the Parquet file to get metadata without loading it
    parquet_file = pq.ParquetFile(parquet_path)
    num_rows = parquet_file.metadata.num_rows

    # Setup for move indexing
    all_possible_moves = get_all_legal_moves()
    move_to_idx = {move: i for i, move in enumerate(all_possible_moves)}

    print(f"Creating empty memory-mapped files for {num_rows} rows...")
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

    start_idx = 0
    iterator = parquet_file.iter_batches(
        batch_size=chunk_size, columns=["fen", "cp", "mate", "line"]
    )

    print(f"Processing data in chunks of {chunk_size}...")
    for batch in tqdm(iterator, total=-(num_rows // -chunk_size)):
        chunk_df = batch.to_pandas()

        board_tensors_chunk = np.array(
            [_get_board_tensor(fen) for fen in chunk_df["fen"]], dtype=np.int8
        )

        conditions = [
            chunk_df["mate"].isna() | (chunk_df["mate"] == 0),
            chunk_df["mate"] > 0,
            chunk_df["mate"] < 0,
        ]
        game_state_choices = [0, 1, 2]
        value_choices = [
            chunk_df["cp"] / 100.0,
            chunk_df["mate"],
            chunk_df["mate"].abs(),
        ]

        game_state_target_chunk = np.select(
            conditions, game_state_choices, default=0
        ).astype(np.int8)
        value_target_chunk = np.select(conditions, value_choices, default=0.0).astype(
            np.float32
        )

        first_moves = chunk_df["line"].apply(_get_first_move_from_line)
        best_move_chunk = np.array(
            [move_to_idx.get(move, -1) for move in first_moves], dtype=np.int64
        )

        end_idx = start_idx + len(chunk_df)
        boards_mm[start_idx:end_idx] = board_tensors_chunk
        states_mm[start_idx:end_idx] = game_state_target_chunk
        values_mm[start_idx:end_idx] = value_target_chunk
        moves_mm[start_idx:end_idx] = best_move_chunk

        start_idx = end_idx

    print("Flushing data to disk...")
    # This ensures all data is written from memory buffers to the disk
    boards_mm.flush()
    states_mm.flush()
    values_mm.flush()
    moves_mm.flush()

    print("Preprocessing complete!")
