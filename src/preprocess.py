# Imports
import numpy as np
import chess
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import os
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

# NEW & OPTIMIZED: Vectorized function to process a whole chunk of FENs
def _get_board_tensors_vectorized(fen_series: pd.Series) -> np.ndarray:
    """Convert a pandas Series of FEN strings to a board tensor (N, 18, 8, 8)."""
    
    num_rows = len(fen_series)
    board_tensors = np.zeros((num_rows, 18, 8, 8), dtype=np.int8)
    
    # --- 1. Piece Placement (Channels 0-11) ---
    piece_to_channel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    }
    
    # Extract piece placement part of FEN
    piece_placements = fen_series.str.split(' ', n=1, expand=True)[0]
    
    # Expand all FENs into a flat 64-char representation
    # e.g., 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR' -> 'rnbqkbnrpppppppp........................PPPPPPPPRNBQKBNR'
    flat_boards = piece_placements.str.replace('/', '').apply(_expand_fen_row)
    
    # Convert the series of strings into a 2D numpy array of characters
    char_board_array = np.array(flat_boards.apply(list).tolist()).reshape(num_rows, 8, 8)
    
    # Use numpy broadcasting to populate piece channels in a vectorized way
    for piece, channel in piece_to_channel.items():
        board_tensors[:, channel, :, :] = (char_board_array == piece)
        
    # --- 2. Side to Move, Castling, En Passant (Channels 12-17) ---
    
    # Channel 12: Side to move (1.0 for White)
    is_white_move = fen_series.str.contains(' w ', regex=False).to_numpy()
    board_tensors[is_white_move, 12, :, :] = 1
    
    # Channels 13-16: Castling rights
    has_w_kingside = fen_series.str.contains('K', regex=False).to_numpy()
    board_tensors[has_w_kingside, 13, :, :] = 1
    
    has_w_queenside = fen_series.str.contains('Q', regex=False).to_numpy()
    board_tensors[has_w_queenside, 14, :, :] = 1
    
    has_b_kingside = fen_series.str.contains('k', regex=False).to_numpy()
    board_tensors[has_b_kingside, 15, :, :] = 1
    
    has_b_queenside = fen_series.str.contains('q', regex=False).to_numpy()
    board_tensors[has_b_queenside, 16, :, :] = 1
    
    # Channel 17: En Passant square (This part is less easily vectorized but still faster)
    ep_squares = fen_series.str.split(' ', n=4, expand=True)[3]
    valid_ep = ep_squares != '-'
    if valid_ep.any():
        ep_indices = ep_squares[valid_ep].index
        ep_coords = ep_squares[valid_ep].apply(lambda s: (chess.SQUARE_NAMES.index(s) // 8, chess.SQUARE_NAMES.index(s) % 8))
        rows, cols = zip(*ep_coords)
        board_tensors[ep_indices, 17, rows, cols] = 1
        
    return board_tensors

# OLD & SLOW: Kept for reference, but not used in the optimized script
def _get_board_tensor_iterative(fen: str) -> np.ndarray:
    """Convert a FEN string to a board tensor (18, 8, 8)."""
    # ... (original function code)
    pass

def preprocess_data_in_chunks(
    parquet_path: str, destination_path: str, chunk_size: int = 100_000
):
    print("Initializing...")
    os.makedirs(destination_path, exist_ok=True)
    
    parquet_file = pq.ParquetFile(parquet_path)
    num_rows = parquet_file.metadata.num_rows
    
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
    
    print(f"Processing data in chunks of {chunk_size} using vectorized operations...")
    for batch in tqdm(iterator, total=-(num_rows // -chunk_size)):
        chunk_df = batch.to_pandas()
        
        # --- OPTIMIZED BOARD TENSOR CREATION ---
        board_tensors_chunk = _get_board_tensors_vectorized(chunk_df["fen"])
        
        # --- OPTIMIZED TARGET CREATION (Your original code was already good here) ---
        conditions = [
            chunk_df["mate"].isna() | (chunk_df["mate"] == 0),
            chunk_df["mate"] > 0,
            chunk_df["mate"] < 0,
        ]
        game_state_choices = [0, 1, 2] # 0: Normal, 1: White Mate, 2: Black Mate
        value_choices = [chunk_df["cp"] / 100.0, chunk_df["mate"], chunk_df["mate"].abs()]
        
        game_state_target_chunk = np.select(
            conditions, game_state_choices, default=0
        ).astype(np.int8)
        value_target_chunk = np.select(conditions, value_choices, default=0.0).astype(
            np.float32
        )
        
        # --- OPTIMIZED MOVE EXTRACTION ---
        first_moves = chunk_df["line"].str.split(' ', n=1, expand=True)[0]
        best_move_chunk = np.array(
            [move_to_idx.get(move, -1) for move in first_moves], dtype=np.int64
        )
        
        # Write chunk to memory-mapped files
        end_idx = start_idx + len(chunk_df)
        boards_mm[start_idx:end_idx] = board_tensors_chunk
        states_mm[start_idx:end_idx] = game_state_target_chunk
        values_mm[start_idx:end_idx] = value_target_chunk
        moves_mm[start_idx:end_idx] = best_move_chunk
        start_idx = end_idx
        
    print("Flushing data to disk...")
    boards_mm.flush()
    states_mm.flush()
    values_mm.flush()
    moves_mm.flush()
    
    print("Preprocessing complete! ✅")