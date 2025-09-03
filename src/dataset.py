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
    def __init__(self, parquet_path, batch_size, start_frac=0.0, end_frac=1.0):
        super().__init__()
        self.parquet_path = parquet_path
        self.batch_size = batch_size

        # Get total rows once for splitting
        pyarrow_dataset = ds.dataset(self.parquet_path, format="parquet")
        self.total_rows = pyarrow_dataset.count_rows()
        self.start_row = int(start_frac * self.total_rows)
        self.end_row = int(end_frac * self.total_rows)
        self.num_rows = self.end_row - self.start_row

        # This mapping is needed for each item, so we create it once
        all_possible_moves = get_all_legal_moves()
        self.move_to_idx = {move: i for i, move in enumerate(all_possible_moves)}

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        chunk_size = self.num_rows // num_workers
        start_row = worker_id * chunk_size
        end_row = start_row + chunk_size
        if worker_id == num_workers - 1:
            end_row = self.num_rows

        pyarrow_dataset = ds.dataset(self.parquet_path, format="parquet")

        # Create a scanner that will read only the required rows
        scanner = pyarrow_dataset.scanner(
            skip_rows=start_row,
            limit=end_row - start_row,
            batch_size=self.batch_size,
            columns={"fen": True, "mate": True, "cp": True, "line": True},
        )

        worker_info = get_worker_info()
        if worker_info is None:  # Single-process data loading
            for batch in scanner.to_batches():
                # Shuffle indices to process rows in random order within the batch
                indices = np.arange(len(batch))
                np.random.shuffle(indices)

                # Convert batch to a more efficient dictionary of lists
                data = batch.to_pydict()

                for i in indices:
                    # Create a row-like dict for the processing function
                    row = {key: val[i] for key, val in data.items()}
                    yield self._process_row(row)
        else:  # Multi-process data loading
            # Each worker will process a subset of batches
            for i, batch in enumerate(scanner.to_batches()):
                if i % worker_info.num_workers == worker_info.id:
                    indices = np.arange(len(batch))
                    np.random.shuffle(indices)
                    data = batch.to_pydict()
                    for idx in indices:
                        row = {key: val[idx] for key, val in data.items()}
                        yield self._process_row(row)

    def __len__(self):
        return self.num_rows

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
