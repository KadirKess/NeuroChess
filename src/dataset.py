# Imports

import torch
from torch.utils.data import IterableDataset, get_worker_info
import numpy as np
import pyarrow.dataset as ds
import chess
import random

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
    def __init__(
        self, parquet_path, start_frac=0.0, end_frac=1.0, shuffle_buffer_size=0
    ):
        super().__init__()
        self.parquet_path = parquet_path
        self.start_frac = start_frac
        self.end_frac = end_frac
        self.shuffle_buffer_size = shuffle_buffer_size

        all_possible_moves = get_all_legal_moves()
        self.move_to_idx = {move: i for i, move in enumerate(all_possible_moves)}

    def _item_generator(self):
        """A generator that yields processed items from the Parquet file."""
        pyarrow_dataset = ds.dataset(self.parquet_path, format="parquet")
        all_batches = list(pyarrow_dataset.to_batches())

        num_batches = len(all_batches)
        start_idx = int(self.start_frac * num_batches)
        end_idx = int(self.end_frac * num_batches)
        target_batches = all_batches[start_idx:end_idx]

        worker_info = get_worker_info()
        if worker_info is None:
            batches_for_this_worker = target_batches
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            batches_for_this_worker = [
                b for i, b in enumerate(target_batches) if i % num_workers == worker_id
            ]

        for batch in batches_for_this_worker:
            rows_as_dicts = batch.to_pylist()
            for row_dict in rows_as_dicts:
                yield self._process_row(row_dict)

    def __iter__(self):
        # If shuffle_buffer_size is 0 or less, don't shuffle.
        if self.shuffle_buffer_size <= 0:
            yield from self._item_generator()
            return

        # Shuffle Buffer Logic
        item_source = self._item_generator()
        buffer = []

        # 1. Fill the buffer initially
        try:
            for _ in range(self.shuffle_buffer_size):
                buffer.append(next(item_source))
        except StopIteration:
            # Dataset is smaller than the buffer size. Just shuffle what we have.
            random.shuffle(buffer)
            yield from buffer
            return

        # 2. Main loop: yield a random item and replace it
        while buffer:
            # Pick a random item to yield
            idx = random.randrange(len(buffer))
            item_to_yield = buffer[idx]

            try:
                # Replace it with the next item from the source
                buffer[idx] = next(item_source)
                yield item_to_yield
            except StopIteration:
                # Source is exhausted. Yield the rest of the buffer.
                # To avoid yielding the same item twice, we swap the picked item
                # with the last one and pop.
                buffer[idx] = buffer[-1]
                buffer.pop()
                yield item_to_yield

    def __len__(self):
        pyarrow_dataset = ds.dataset(self.parquet_path, format="parquet")
        total_rows = pyarrow_dataset.count_rows()

        # Calculate the number of rows in the subset for this instance
        start_row = int(self.start_frac * total_rows)
        end_row = int(self.end_frac * total_rows)

        return end_row - start_row

    def _process_row(self, row):

        board_tensor = _get_board_tensor(row["fen"])
        mate = row["mate"]
        cp = row["cp"]

        if isinstance(mate, str):
            if mate.startswith("#"):
                mate = mate[1:]
        try:
            mate = float(mate)
        except (ValueError, TypeError):
            mate = np.nan

        if mate == 0.0 or np.isnan(mate):
            game_state = 0  # Normal
            value = cp / 100.0
        elif mate > 0:
            game_state = 1  # Mate in X for white
            value = 1.0
        else:
            game_state = 2  # Mate in X for black
            value = -1.0

        # Get policy tensor
        policy_tensor = torch.zeros(len(self.move_to_idx))
        move_idx = self.move_to_idx.get(row["move"], -1)
        if move_idx != -1:
            policy_tensor[move_idx] = 1.0

        return board_tensor, game_state, value, policy_tensor
