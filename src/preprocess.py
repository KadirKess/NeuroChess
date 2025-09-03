# Imports

import sys
import numpy as np
import chess
import pandas as pd

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


def preprocess_data(parquet_path: str, destination_path: str):
    print("Loading parquet file...")
    df = pd.read_parquet(parquet_path, columns=["fen", "cp", "mate", "line"])

    all_possible_moves = get_all_legal_moves()

    move_to_idx = {move: i for i, move in enumerate(all_possible_moves)}
    num_possible_moves = len(all_possible_moves)

    print("Processing board tensors...")
    board_tensors = np.array([_get_board_tensor(fen) for fen in df["fen"]])

    print("Processing labels...")

    conditions = [df["mate"].isna() | (df["mate"] == 0), df["mate"] > 0, df["mate"] < 0]
    game_state_choices = [0, 1, 2]

    value_choices = [df["cp"] / 100.0, df["mate"], df["mate"].abs()]

    game_state_target = np.select(conditions, game_state_choices, default=0).astype(
        np.int64
    )
    value_target = np.select(conditions, value_choices, default=0.0).astype(np.float32)

    first_moves = df["line"].apply(_get_first_move_from_line)
    best_move = np.array(
        [move_to_idx.get(move, -1) for move in first_moves], dtype=np.int64
    )

    print("Saving processed data to disk...")

    import os

    os.makedirs(destination_path, exist_ok=True)

    np.save(f"{destination_path}/board_tensors.npy", board_tensors)
    np.save(f"{destination_path}/game_state_target.npy", game_state_target)
    np.save(f"{destination_path}/value_target.npy", value_target)
    np.save(f"{destination_path}/best_move.npy", best_move)

    print("Preprocessing complete.")
