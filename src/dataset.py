# Imports

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import chess

# Local imports

from src.all_moves import get_all_legal_moves

# Dataset class for chess positions


class PositionsDataset(Dataset):
    def __init__(self, parquet_path: str):
        self.df = pd.read_parquet(parquet_path, columns=["fen", "cp", "mate", "line"])

        self._all_possible_moves = get_all_legal_moves()

        self.move_to_idx = {move: i for i, move in enumerate(self._all_possible_moves)}
        self.num_possible_moves = len(self._all_possible_moves)

    def _get_first_move_from_line(self, line: str) -> str:
        """
        Extract the first move from a line of moves.
        """
        moves = line.split()
        if moves:
            return moves[0]
        return ""

    def _get_board_tensor(self, fen: str) -> np.ndarray:
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get a chess position by index.
        """

        # Get the row we are interested in
        row = self.df.iloc[idx]

        # Process the board
        board_tensor = self._get_board_tensor(row["fen"])

        # Process the evaluation and mate in
        mate_in = row.get("mate")  # Use .get() for safety
        if pd.isna(mate_in) or mate_in == 0:
            game_state_target = 0  # Normal
            value_target = row["cp"] / 100.0  # Convert to centipawns
        elif mate_in > 0:
            game_state_target = 1  # White Mate
            value_target = mate_in
        else:  # mate_in < 0
            game_state_target = 2  # Black Mate
            value_target = abs(mate_in)

        # Find the best move
        first_move = self._get_first_move_from_line(row["line"])
        best_move = self.move_to_idx.get(first_move, -1)

        # If the best move is not found, throw an error
        if best_move == -1:
            raise ValueError(
                f"Best move not found for FEN: {row['fen']}. The parsed first move is '{first_move}'"
            )

        return {
            "board_tensor": torch.from_numpy(board_tensor),
            "game_state_target": torch.tensor(game_state_target, dtype=torch.long),
            "value_target": torch.tensor(value_target, dtype=torch.float32),
            "best_move": torch.tensor(best_move, dtype=torch.long),
        }
