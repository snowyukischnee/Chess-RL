from typing import Any
import numpy as np
import chess


class ChessEnv(object):
    """
    An implementation of chess environment
    """

    def __init__(self) -> None:
        self.board = None
        self.actions = ChessEnv.init_actions()
        self.n_actions = len(self.actions)
        self.n_state = (12, 8, 8)

    @staticmethod
    def init_actions() -> list:
        """Generate a list of moves which contains some illegal moves
        :return:
            list: list of uci strings
        """
        _letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        _numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
        _promoted = ['q', 'r', 'b', 'n']  # queen, rook, bishop, knight
        _actions = []
        for l_o in range(8):
            for n_o in range(8):
                _destinations = [(l_o, x) for x in range(8)] + \
                                [(x, n_o) for x in range(8)] + \
                                [(l_o + x, n_o + x) for x in range(-7, 8)] + \
                                [(l_o + x, n_o - x) for x in range(-7, 8)] + \
                                [(l_o + x, n_o + y) for (x, y) in
                                 [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (1, -2), (2, 1), (1, 2)]]
                for (l_d, n_d) in _destinations:
                    if (l_o, n_o) != (l_d, n_d) and l_d in range(8) and n_d in range(8):
                        _actions.append(_letters[l_o] + _numbers[n_o] + _letters[l_d] + _numbers[n_d])
        for l_o in range(8):
            for p in _promoted:  # move forward and promote
                _actions.append(_letters[l_o] + _numbers[1] + _letters[l_o] + _numbers[0] + p)
                _actions.append(_letters[l_o] + _numbers[6] + _letters[l_o] + _numbers[7] + p)
                if l_o > 0:  # kill a piece and promote
                    _actions.append(_letters[l_o] + _numbers[1] + _letters[l_o - 1] + _numbers[0] + p)
                    _actions.append(_letters[l_o] + _numbers[6] + _letters[l_o - 1] + _numbers[7] + p)
                if l_o < 7:  # kill a piece and promote
                    _actions.append(_letters[l_o] + _numbers[1] + _letters[l_o + 1] + _numbers[0] + p)
                    _actions.append(_letters[l_o] + _numbers[6] + _letters[l_o + 1] + _numbers[7] + p)
        return _actions

    def board2state(self) -> Any:
        """Convert the board to state
        :param
            board(chess.Board): the chessboard class
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
        _board_overview[0] = self.turn
        _board_overview[1] = self.board.has_kingside_castling_rights(chess.WHITE)
        _board_overview[2] = self.board.has_queenside_castling_rights(chess.WHITE)
        _board_overview[3] = self.board.has_kingside_castling_rights(chess.BLACK)
        _board_overview[4] = self.board.has_queenside_castling_rights(chess.BLACK)
        _board_overview[5] = len(self.board.pieces(chess.PAWN, chess.WHITE))
        _board_overview[6] = len(self.board.pieces(chess.ROOK, chess.WHITE))
        _board_overview[7] = len(self.board.pieces(chess.KNIGHT, chess.WHITE))
        _board_overview[8] = len(self.board.pieces(chess.BISHOP, chess.WHITE))
        _board_overview[9] = len(self.board.pieces(chess.QUEEN, chess.WHITE))
        _board_overview[10] = len(self.board.pieces(chess.KING, chess.WHITE))
        _board_overview[11] = len(self.board.pieces(chess.PAWN, chess.BLACK))
        _board_overview[12] = len(self.board.pieces(chess.ROOK, chess.BLACK))
        _board_overview[13] = len(self.board.pieces(chess.KNIGHT, chess.BLACK))
        _board_overview[14] = len(self.board.pieces(chess.BISHOP, chess.BLACK))
        _board_overview[15] = len(self.board.pieces(chess.QUEEN, chess.BLACK))
        _board_overview[16] = len(self.board.pieces(chess.KING, chess.BLACK))
        _legal_actions = np.zeros(shape=len(self.actions), dtype=np.int32)
        for i in range(len(self.actions)):
            if chess.Move.from_uci(self.actions[i]) in self.board.legal_moves:
                _legal_actions[i] = 1
        _piece_positions = np.zeros(shape=(12, 64), dtype=np.int32)
        _piece_positions[0][list(game.board.pieces(chess.PAWN, chess.WHITE))] = 1
        _piece_positions[1][list(game.board.pieces(chess.ROOK, chess.WHITE))] = 1
        _piece_positions[2][list(game.board.pieces(chess.KNIGHT, chess.WHITE))] = 1
        _piece_positions[3][list(game.board.pieces(chess.BISHOP, chess.WHITE))] = 1
        _piece_positions[4][list(game.board.pieces(chess.QUEEN, chess.WHITE))] = 1
        _piece_positions[5][list(game.board.pieces(chess.KING, chess.WHITE))] = 1
        _piece_positions[6][list(game.board.pieces(chess.PAWN, chess.BLACK))] = 1
        _piece_positions[7][list(game.board.pieces(chess.ROOK, chess.BLACK))] = 1
        _piece_positions[8][list(game.board.pieces(chess.KNIGHT, chess.BLACK))] = 1
        _piece_positions[9][list(game.board.pieces(chess.BISHOP, chess.BLACK))] = 1
        _piece_positions[10][list(game.board.pieces(chess.QUEEN, chess.BLACK))] = 1
        _piece_positions[11][list(game.board.pieces(chess.KING, chess.BLACK))] = 1
        _attack_map = np.zeros(shape=(128, 64), dtype=np.int32)
        for i in range(0, 64):
            _attack_map[i][list(self.board.attackers(chess.WHITE, i))] = 1
        for i in range(64, 128):
            _attack_map[i][list(self.board.attackers(chess.BLACK, i))] = 1
        return _board_overview, _legal_actions, _piece_positions

    @property
    def turn(self) -> bool:
        """Current turn of the board
        :return:
            bool: current turn. True for current turn is White's turn, False is Black's turn
        """
        return self.board.turn == chess.WHITE

    @property
    def done(self) -> bool:
        """Current board's done status
        :return:
            bool: board's done status. True for the game is done, False otherwise
        """
        return self.board.result(claim_draw=True) != '*'
    
    @property
    def result(self) -> Any:
        """Get the game's result if done
        :return:
            Any: game's result. None for game is undone, str for game is done
        """
        if self.board.result(claim_draw=True) != '*':
            return self.board.result(claim_draw=True)
        else:
            return None

    def reset(self, fen: str = None) -> Any:
        """Reset the game
        :param
            fen(str): the initial configuration for the board. None for default setting
        :return:
            numpy.ndarray: the board's current state
        """
        if fen:
            self.board = chess.Board(fen=fen)
        else:
            self.board = chess.Board()
        return self.board2state()

    def step(self, action: int) -> Any:
        """Make an action in the environment
        :param
            action(int): choose the action
        :return:
            tuple(tuple of np.ndarray, float, bool, Any): next_state, reward, done, info. Following openai-gym's env
            if move is illegal then reward is -0.5
            reward is 1 if win, -1 if lose. -0.5 if draw
        """
        _action = self.actions[action]
        _reward = 0
        _legal_move = chess.Move.from_uci(_action) in self.board.legal_moves
        _current_turn = self.turn
        if _legal_move:
            self.board.push_uci(_action)
        else:
            _reward = 0
        _next_state = self.board2state()
        if self.done:
            if self.result == '1-0':
                if _current_turn:
                    _reward = 1
                else:
                    _reward = -1
            elif self.result == '0-1':
                if _current_turn:
                    _reward = -1
                else:
                    _reward = 1
            else:
                _reward = 0
        return _next_state, _reward, self.done, None


if __name__ == '__main__':
    """For testing purpose only
    """
    game = ChessEnv()
    state = game.reset(fen='rnbqkbnr/ppp3pp/2p3p1/2p5/8/5Q2/PPPPPPPP/RNB1KBNR w KQkq -')
    game.board.push_uci('f3e3')
    print(game.board.is_check())
    print(game.board.was_into_check())
    # for action in game.actions:
    #     if chess.Move.from_uci(action) in game.board.legal_moves:
    #         print(action)
    # game.board.push_uci('d8e7')
    # game.board.push_uci('c7c8q')
    # print(r, d, game.result)
    print(game.board)
    # print(game.board.attackers(chess.WHITE, chess.D2))
    x = np.zeros(shape=(12, 64), dtype=np.int32)
    x[0][list(game.board.pieces(chess.PAWN, chess.WHITE))] = 1
    print(x[0])
    print(list(game.board.attackers(chess.WHITE, 11)))
    # is_check() after move then if current side still check
    # was_into_check() the move check or not
    # castling_rights
    # turn
    # attacks(square)
