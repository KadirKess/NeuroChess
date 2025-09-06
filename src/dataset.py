# Imports

import torch
from torch.utils.data import IterableDataset, get_worker_info
import numpy as np
import pyarrow.parquet as pq
import chess
import random
from typing import List

# Local imports

from src.all_moves import get_all_legal_moves


# Helper function to expand a single row of a FEN's piece placement section
def _expand_fen_row(row_str: str) -> str:
    expanded = ""
    for char in row_str:
        if char.isdigit():
            expanded += "." * int(char)
        else:
            expanded += char
    return expanded


# Vectorized function to process a whole chunk of FENs
def _get_board_tensor(fen: str) -> np.ndarray:
    """Convert a FEN string to a board tensor (18, 8, 8)."""
    board_tensor = np.zeros((18, 8, 8), dtype=np.int8)

    parts = fen.split(" ")
    piece_placement = parts[0]
    side_to_move = parts[1]
    castling = parts[2]
    en_passant = parts[3]

    # 1. Piece Placement (Channels 0-11)
    piece_to_channel = {
        "P": 0,
        "N": 1,
        "B": 2,
        "R": 3,
        "Q": 4,
        "K": 5,
        "p": 6,
        "n": 7,
        "b": 8,
        "r": 9,
        "q": 10,
        "k": 11,
    }
    rows = piece_placement.split("/")
    for r, row_str in enumerate(rows):
        c = 0
        for char in row_str:
            if char.isdigit():
                c += int(char)
            else:
                board_tensor[piece_to_channel[char], r, c] = 1
                c += 1

    # 2. Side to move (Channel 12)
    if side_to_move == "w":
        board_tensor[12, :, :] = 1

    # 3. Castling rights (Channels 13-16)
    if "K" in castling:
        board_tensor[13, :, :] = 1
    if "Q" in castling:
        board_tensor[14, :, :] = 1
    if "k" in castling:
        board_tensor[15, :, :] = 1
    if "q" in castling:
        board_tensor[16, :, :] = 1

    # 4. En Passant square (Channel 17)
    if en_passant != "-":
        ep_square = chess.SQUARE_NAMES.index(en_passant)
        row, col = ep_square // 8, ep_square % 8
        board_tensor[17, row, col] = 1

    return board_tensor


class IterablePositionsDataset(IterableDataset):
    def __init__(self, parquet_paths: List[str], shuffle_buffer_size: int = 0):
        super().__init__()
        self.parquet_paths = parquet_paths
        self.shuffle_buffer_size = shuffle_buffer_size

        total_rows = 0
        for path in self.parquet_paths:
            pq_file = pq.ParquetFile(path)
            total_rows += pq_file.metadata.num_rows
        self.num_rows = total_rows

        # It's recommended to pre-compute this list and save it as a constant
        # in a separate file (e.g., `src/legal_moves.py`) to avoid re-computing it.
        all_possible_moves = get_all_legal_moves()
        self.move_to_idx = {move: i for i, move in enumerate(all_possible_moves)}

    def _process_batch(self, batch_data):
        """Processes a batch of rows from the Parquet file into tensors."""
        fens = batch_data["fen"]
        n_samples = len(fens)

        # 1. Batch process FENs to board tensors
        board_tensors = np.array([_get_board_tensor(fen) for fen in fens])

        # 2. Vectorize target creation using NumPy
        # In pandas/pyarrow, 'None' becomes NaN for numeric types
        mates = np.array(batch_data["mate"], dtype=np.float32)
        cps = np.array(batch_data["cp"], dtype=np.float32)

        game_states = np.zeros(n_samples, dtype=np.int64)
        values = np.zeros(n_samples, dtype=np.float32)

        # Create boolean masks for each game state
        is_normal = np.isnan(mates) | (mates == 0)
        is_white_mate = mates > 0
        is_black_mate = mates < 0

        # Apply conditions using masks for vectorization
        game_states[is_normal] = 0
        values[is_normal] = cps[is_normal] / 100.0

        game_states[is_white_mate] = 1
        values[is_white_mate] = mates[is_white_mate]

        game_states[is_black_mate] = 2
        values[is_black_mate] = np.abs(mates[is_black_mate])

        # 3. Batch process best moves
        lines = batch_data["line"]
        first_moves = [line.split(" ")[0] if line else "" for line in lines]
        best_move_indices = np.array(
            [self.move_to_idx.get(move, -1) for move in first_moves], dtype=np.int64
        )

        # Create a list of dictionaries to be yielded
        processed_items = []
        for i in range(n_samples):
            processed_items.append(
                {
                    "board_tensor": torch.from_numpy(board_tensors[i]).float(),
                    "game_state_target": torch.tensor(game_states[i], dtype=torch.long),
                    "value_target": torch.tensor(values[i], dtype=torch.float32),
                    "best_move": torch.tensor(best_move_indices[i], dtype=torch.long),
                }
            )
        return processed_items

    def _item_generator(self):
        """A generator that yields processed items from all Parquet files."""
        worker_info = get_worker_info()
        batch_idx = -1

        for path in self.parquet_paths:
            pq_file = pq.ParquetFile(path)
            for rg in range(pq_file.num_row_groups):
                for batch in pq_file.iter_batches(batch_size=2048, row_groups=[rg]):
                    batch_idx += 1

                    # Distribute batch processing among workers
                    if worker_info and (
                        batch_idx % worker_info.num_workers != worker_info.id
                    ):
                        continue

                    data = batch.to_pydict()

                    # Process the entire batch at once
                    processed_batch = self._process_batch(data)

                    # Shuffle within the processed batch before yielding
                    random.shuffle(processed_batch)

                    for item in processed_batch:
                        yield item

    def __iter__(self):
        if self.shuffle_buffer_size > 0:
            # Implement shuffle buffer
            buffer = []
            item_gen = self._item_generator()

            # Initially fill the buffer
            try:
                for _ in range(self.shuffle_buffer_size):
                    buffer.append(next(item_gen))
            except StopIteration:
                # If dataset is smaller than buffer, just shuffle and yield
                random.shuffle(buffer)
                for item in buffer:
                    yield item
                return

            # Stream the rest of the data, replacing items in the buffer
            for item in item_gen:
                idx_to_yield = random.randint(0, self.shuffle_buffer_size - 1)
                yield buffer[idx_to_yield]
                buffer[idx_to_yield] = item

            # Yield remaining items in the buffer
            random.shuffle(buffer)
            for item in buffer:
                yield item
        else:
            # No shuffling, just yield directly
            yield from self._item_generator()

    def __len__(self):
        return self.num_rows
