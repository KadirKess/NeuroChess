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
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
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

        all_possible_moves = get_all_legal_moves()
        self.move_to_idx = {move: i for i, move in enumerate(all_possible_moves)}

    def _row_generator(self):
        """A generator that yields processed rows from all Parquet files."""
        worker_info = get_worker_info()
        batch_idx = -1

        for path in self.parquet_paths:
            pq_file = pq.ParquetFile(path)
            for rg in range(pq_file.num_row_groups):
                for batch in pq_file.iter_batches(batch_size=2048, row_groups=[rg]):
                    batch_idx += 1

                    # Distribute batches among workers
                    if worker_info and (batch_idx % worker_info.num_workers != worker_info.id):
                        continue

                    data = batch.to_pydict()
                    indices = np.arange(len(batch))
                    np.random.shuffle(indices) # Shuffle within the batch

                    for i in indices:
                        row = {key: val[i] for key, val in data.items()}
                        yield self._process_row(row)

    def __iter__(self):
        if self.shuffle_buffer_size > 0:
            # Implement shuffle buffer
            buffer = []
            row_gen = self._row_generator()

            # Initially fill the buffer
            try:
                for _ in range(self.shuffle_buffer_size):
                    buffer.append(next(row_gen))
            except StopIteration:
                # If dataset is smaller than buffer, just shuffle and yield
                random.shuffle(buffer)
                for item in buffer:
                    yield item
                return

            # Stream the rest of the data, replacing items in the buffer
            for item in row_gen:
                idx_to_yield = random.randint(0, self.shuffle_buffer_size - 1)
                yield buffer[idx_to_yield]
                buffer[idx_to_yield] = item

            # Yield remaining items in the buffer
            random.shuffle(buffer)
            for item in buffer:
                yield item
        else:
            # No shuffling, just yield directly
            yield from self._row_generator()

    def __len__(self):
        return self.num_rows

    def _process_row(self, row):
        """Processes a single row from the Parquet file into tensors."""
        board_tensor = _get_board_tensor(row["fen"])

        mate = row["mate"]
        cp = row["cp"]

        if mate is None or mate == 0.0:
            game_state = 0  # Normal
            value = cp / 100.0
        elif mate > 0:
            game_state = 1  # White Mate
            value = mate
        else:  # mate < 0
            game_state = 2  # Black Mate
            value = abs(mate)

        first_move = row["line"].split(" ")[0]
        best_move_idx = self.move_to_idx.get(first_move, -1)

        return {
            "board_tensor": torch.from_numpy(board_tensor).float(),
            "game_state_target": torch.tensor(game_state, dtype=torch.long),
            "value_target": torch.tensor(value, dtype=torch.float32),
            "best_move": torch.tensor(best_move_idx, dtype=torch.long),
        }