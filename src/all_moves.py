# Imports

import chess


def get_all_legal_moves():

    # Create an empty chess board
    board = chess.Board()
    board.clear()

    # Generate all legal moves
    legal_moves = set()

    # 1. We add all the moves a queen and a knight can make
    for square in chess.SQUARES:  # Iterate over all squares
        for piece in [chess.QUEEN, chess.KNIGHT]:  # Limit to specific pieces
            board.set_piece_at(square, chess.Piece(piece, chess.WHITE))
            for move in board.legal_moves:
                legal_moves.add(move.uci())
            board.remove_piece_at(square)

    # 2. Pawn promotions
    for from_file in range(8):
        # White promotions
        from_square_w = chess.square(from_file, 6)
        for to_file_diff in [-1, 0, 1]:
            to_file_w = from_file + to_file_diff
            if 0 <= to_file_w < 8:
                to_square_w = chess.square(to_file_w, 7)
                for piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    legal_moves.add(
                        chess.Move(from_square_w, to_square_w, promotion=piece).uci()
                    )

        # Black promotions
        from_square_b = chess.square(from_file, 1)
        for to_file_diff in [-1, 0, 1]:
            to_file_b = from_file + to_file_diff
            if 0 <= to_file_b < 8:
                to_square_b = chess.square(to_file_b, 0)
                for piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    legal_moves.add(
                        chess.Move(from_square_b, to_square_b, promotion=piece).uci()
                    )

    return sorted(list(legal_moves))
