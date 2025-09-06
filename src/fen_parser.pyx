import numpy as np
cimport numpy as np

cdef dict piece_to_channel = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
}

cpdef np.ndarray[np.int8_t, ndim=3] parse_fen(str fen):
    """
    A Cython-optimized function to convert a FEN string to a board tensor.
    """
    # Declare C-typed variables
    cdef np.ndarray[np.int8_t, ndim=3] board_tensor = np.zeros((18, 8, 8), dtype=np.int8)
    cdef int r, c, piece_val
    cdef str char

    parts = fen.split(" ")
    piece_placement = parts[0]
    side_to_move = parts[1]
    castling = parts[2]
    en_passant = parts[3]

    # 1. Piece Placement (Channels 0-11)
    r = 0
    for row_str in piece_placement.split('/'):
        c = 0
        for char in row_str:
            if '1' <= char <= '8':
                c += int(char)
            else:
                piece_val = piece_to_channel[char]
                board_tensor[piece_val, r, c] = 1
                c += 1
        r += 1

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
        col = ord(en_passant[0]) - ord('a')
        row = int(en_passant[1]) - 1
        board_tensor[17, row, col] = 1
        
    return board_tensor