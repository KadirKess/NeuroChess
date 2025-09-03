import torch
import chess
import numpy as np
import argparse
import os

# Local imports from your project structure
from src.model import ChessResNetModel
from src.dataset import _get_board_tensor
from src.all_moves import get_all_legal_moves


def load_model(model_path: str, device: torch.device) -> ChessResNetModel:
    """
    Loads the trained ChessResNetModel from the specified path.
    """
    model = ChessResNetModel()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Load the saved state dictionary, mapping it to the correct device
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    print("🤖 Model loaded successfully.")
    return model


def choose_ai_move(
    model: ChessResNetModel,
    board: chess.Board,
    device: torch.device,
    move_to_idx: dict,
    idx_to_move: dict,
) -> chess.Move:
    """
    Analyzes the position and chooses the best move for the AI.

    This function performs three key steps:
    1.  Converts the current board state into a tensor.
    2.  Gets the model's policy output (logits for all possible moves).
    3.  Masks out illegal moves and selects the best legal move.
    """
    # 1. Preprocess the current board state into a tensor
    fen = board.fen()
    board_tensor = _get_board_tensor(fen)
    board_tensor = torch.from_numpy(board_tensor).float().unsqueeze(0).to(device)

    with torch.no_grad():
        # 2. Get model's prediction
        outputs = model(board_tensor)
        # We only need the policy head's output
        policy_logits = outputs["best_move"].squeeze(0)

        # 3. Create a mask to filter out illegal moves
        mask = torch.full(policy_logits.shape, -float('inf'), device=device)
        
        legal_move_indices = [
            move_to_idx[move.uci()] for move in board.legal_moves if move.uci() in move_to_idx
        ]
        
        if legal_move_indices:
            mask[legal_move_indices] = 0.0

        # 4. Apply the mask and find the best legal move
        masked_logits = policy_logits + mask
        best_move_idx = torch.argmax(masked_logits).item()
        
        # 5. Convert the chosen move index back to a chess.Move object
        best_move_uci = idx_to_move[best_move_idx]
        return chess.Move.from_uci(best_move_uci)


def play_game(model: ChessResNetModel, device: torch.device):
    """
    Main game loop for playing against the AI.
    """
    # Create the mappings from UCI move strings to tensor indices
    all_moves_list = get_all_legal_moves()
    move_to_idx = {move: i for i, move in enumerate(all_moves_list)}
    idx_to_move = {i: move for i, move in enumerate(all_moves_list)}
    
    board = chess.Board()
    
    # Let the user choose their color
    player_color = None
    while player_color is None:
        choice = input("Do you want to play as White (w) or Black (b)? ").lower()
        if choice in ['white', 'w']:
            player_color = chess.WHITE
        elif choice in ['black', 'b']:
            player_color = chess.BLACK
        else:
            print("Invalid choice. Please enter 'w' or 'b'.")
            
    # Main game loop
    while not board.is_game_over():
        print("\n" + "="*20)
        print(board)
        print("="*20 + "\n")

        if board.turn == player_color:
            # Player's turn
            while True:
                try:
                    move_uci = input("Enter your move (e.g., e2e4): ")
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("⚠️ Illegal move. Try again.")
                except ValueError:
                    print("⚠️ Invalid move format. Use UCI format (e.g., e2e4).")
        else:
            # AI's turn
            print("AI is thinking...")
            ai_move = choose_ai_move(model, board, device, move_to_idx, idx_to_move)
            print(f"🤖 AI plays: {ai_move.uci()}")
            board.push(ai_move)

    # Game over
    print("\n" + "="*20)
    print("🎉 GAME OVER 🎉")
    print(f"Result: {board.result()}")
    print("Final board:")
    print(board)
    print("="*20 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Play chess against a trained model.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=".",
        help="Directory where the 'best_model.pth' is saved. Defaults to the current directory.",
    )
    args = parser.parse_args()

    # Inference on CPU is generally sufficient for a single position
    device = torch.device("cpu")
    model_path = os.path.join(args.save_dir, "best_model.pth")
    
    try:
        model = load_model(model_path, device)
        play_game(model, device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have trained the model and 'best_model.pth' exists in the specified directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()