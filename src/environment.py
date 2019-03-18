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

    @staticmethod
    def board2state(board: chess.Board) -> np.ndarray:
        """Convert the board to 3-D numpy array
        :param
            board(chess.Board): the chessboard class
        :return:
            numpy.ndarray: 3-D numpy array which sized (12,8,8) represent the board which is sparse and contain only 1 and 0
        """
        _state = np.zeros(shape=(12, 8, 8), dtype=np.int32)
        _lines = board.__str__().split('\n')
        _dict = {'r': 0, 'n': 1, 'b': 2, 'q': 3, 'k': 4, 'p': 5, 'R': 6, 'N': 7, 'B': 8, 'Q': 9, 'K': 10, 'P': 11}
        for i in range(8):
            for j in range(8):
                if _lines[i][j * 2] != '.':
                    _state[_dict[_lines[i][j * 2]]][i][j] = 1
        return _state

    @property
    def turn(self) -> bool:
        """Current turn of the board
        :return:
            bool: current turn. True for current turn is White's turn, False is Black's turn
        """
        return self.board.turn

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

    def reset(self, fen: str = None) -> np.ndarray:
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
        return ChessEnv.board2state(self.board)

    def step(self, action: int) -> Any:
        """Make an action in the environment
        :param
            action(int): choose the action
        :return:
            tuple(numpy.ndarray, float, bool, Any): next_state, reward, done, info. Following openai-gym's env
            if move is illegal then reward is -0.5
            reward is 1 if win, -1 if lose. -0.5 if draw
        """
        _action = self.actions[action]
        _next_state = np.zeros(shape=(12, 8, 8), dtype=np.int32)
        _reward = 0
        _legal_move = chess.Move.from_uci(_action) in self.board.legal_moves
        _current_turn = self.turn
        if _legal_move:
            self.board.push_uci(_action)
            _next_state = ChessEnv.board2state(self.board)
        else:
            _reward = -0.5
            _next_state = ChessEnv.board2state(self.board)
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
                _reward = -0.5
        return _next_state, _reward, self.done, None


if __name__ == '__main__':
    """For testing purpose only
    """
    game = ChessEnv()
    state = game.reset()
    ns, r, d, _ = game.step(928)
    print(r, d, game.result)
    print(game.board)
    print(len(game.actions))