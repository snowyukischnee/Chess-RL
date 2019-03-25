from typing import Any
import numpy as np
import chess


def board2state0(board: chess.Board, actions: list) -> Any:
    """Convert the board to state
    :param
        board(chess.Board): the chessboard class
        actions(list): list all available action (including some illegal moves)
    :return:
        numpy.ndarray(17): [
        // turn
        is_white_turn,
        // castling rights
        castling_rights_white_king_side,
        castling_rights_white_queen_side,
        castling_rights_black_king_size,
        castling_rights_black_queen_size,
        // number of each pieces
        number_of_white_pawn,
        number_of_white_rook,
        number_of_white_knight,
        number_of_white_bishop,
        number_of_white_queen,
        number_of_white_king,
        number_of_black_pawn,
        number_of_black_rook,
        number_of_black_knight,
        number_of_black_bishop,
        number_of_black_queen,
        number_of_black_king,
        ],
        numpy.ndarray(len(actions)): [1 or 0 indicate i-th action is legal],
        numpy.ndarray(12,64): represent board,
        numpy.ndarray(128,64): 2 attack map of white and black
    """
    _board_overview = np.zeros(shape=17, dtype=np.int32)
    _board_overview[0] = board.turn
    _board_overview[1] = board.has_kingside_castling_rights(chess.WHITE)
    _board_overview[2] = board.has_queenside_castling_rights(chess.WHITE)
    _board_overview[3] = board.has_kingside_castling_rights(chess.BLACK)
    _board_overview[4] = board.has_queenside_castling_rights(chess.BLACK)
    _board_overview[5] = len(board.pieces(chess.PAWN, chess.WHITE))
    _board_overview[6] = len(board.pieces(chess.ROOK, chess.WHITE))
    _board_overview[7] = len(board.pieces(chess.KNIGHT, chess.WHITE))
    _board_overview[8] = len(board.pieces(chess.BISHOP, chess.WHITE))
    _board_overview[9] = len(board.pieces(chess.QUEEN, chess.WHITE))
    _board_overview[10] = len(board.pieces(chess.KING, chess.WHITE))
    _board_overview[11] = len(board.pieces(chess.PAWN, chess.BLACK))
    _board_overview[12] = len(board.pieces(chess.ROOK, chess.BLACK))
    _board_overview[13] = len(board.pieces(chess.KNIGHT, chess.BLACK))
    _board_overview[14] = len(board.pieces(chess.BISHOP, chess.BLACK))
    _board_overview[15] = len(board.pieces(chess.QUEEN, chess.BLACK))
    _board_overview[16] = len(board.pieces(chess.KING, chess.BLACK))
    _legal_actions = np.zeros(shape=len(actions), dtype=np.int32)
    for i in range(len(actions)):
        if chess.Move.from_uci(actions[i]) in board.legal_moves:
            _legal_actions[i] = 1
    _piece_positions = np.zeros(shape=(12, 64), dtype=np.int32)
    _piece_positions[0][list(board.pieces(chess.PAWN, chess.WHITE))] = 1
    _piece_positions[1][list(board.pieces(chess.ROOK, chess.WHITE))] = 1
    _piece_positions[2][list(board.pieces(chess.KNIGHT, chess.WHITE))] = 1
    _piece_positions[3][list(board.pieces(chess.BISHOP, chess.WHITE))] = 1
    _piece_positions[4][list(board.pieces(chess.QUEEN, chess.WHITE))] = 1
    _piece_positions[5][list(board.pieces(chess.KING, chess.WHITE))] = 1
    _piece_positions[6][list(board.pieces(chess.PAWN, chess.BLACK))] = 1
    _piece_positions[7][list(board.pieces(chess.ROOK, chess.BLACK))] = 1
    _piece_positions[8][list(board.pieces(chess.KNIGHT, chess.BLACK))] = 1
    _piece_positions[9][list(board.pieces(chess.BISHOP, chess.BLACK))] = 1
    _piece_positions[10][list(board.pieces(chess.QUEEN, chess.BLACK))] = 1
    _piece_positions[11][list(board.pieces(chess.KING, chess.BLACK))] = 1
    _attack_map = np.zeros(shape=(128, 64), dtype=np.int32)
    for i in range(0, 64):
        _attack_map[i][list(board.attackers(chess.WHITE, i))] = 1
    for i in range(64, 128):
        _attack_map[i][list(board.attackers(chess.BLACK, i - 64))] = 1
    return _board_overview, _legal_actions, _piece_positions, _attack_map
