# Imports

import torch
from torch.utils.data import IterableDataset, get_worker_info
import numpy as np
import pyarrow.dataset as ds
import chess

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
    def __init__(self, parquet_path, start_frac=0.0, end_frac=1.0):
        super().__init__()
        self.parquet_path = parquet_path
        self.start_frac = start_frac
        self.end_frac = end_frac

        # This mapping is needed for each item, so we create it once
        all_possible_moves = get_all_legal_moves()
        self.move_to_idx = {move: i for i, move in enumerate(all_possible_moves)}

    def __iter__(self):
        # Create a pyarrow dataset - this is memory-efficient
        pyarrow_dataset = ds.dataset(self.parquet_path, format="parquet")

        # Get all batches (row groups) from the dataset
        all_batches = list(pyarrow_dataset.to_batches())

        # Determine the subset of batches for this dataset instance (for train/val split)
        num_batches = len(all_batches)
        start_idx = int(self.start_frac * num_batches)
        end_idx = int(self.end_frac * num_batches)
        target_batches = all_batches[start_idx:end_idx]

        # Distribute work among workers
        worker_info = get_worker_info()
        if worker_info is None:
            # Single-process data loading, this process handles all its target batches
            batches_for_this_worker = target_batches
        else:
            # Multi-process data loading, split the target batches among workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            batches_for_this_worker = [
                b for i, b in enumerate(target_batches) if i % num_workers == worker_id
            ]

        # Process and yield each row from the assigned batches
        for batch in batches_for_this_worker:
            df = batch.to_pandas()
            for _, row in df.iterrows():
                yield self._process_row(row)

    def _process_row(self, row):
        """Processes a single row from the Parquet file into tensors."""
        board_tensor = _get_board_tensor(row["fen"])

        mate = row["mate"]
        cp = row["cp"]

        if mate == 0.0 or np.isnan(mate):
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
