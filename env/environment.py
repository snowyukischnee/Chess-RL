from typing import Any
import numpy as np
import chess

import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../')))
from CP_CHESS.utils.board2state import Board2State


class ChessEnv(object):
    """
    An implementation of chess environment
    """

    def __init__(self) -> None:
        self.board = None
        self.actions = ChessEnv.init_actions()
        self.done = False
        self.result = None

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
                                 [(-2, -1), (-1, -2), (-2, 1), (-1, 2), (2, -1), (1, -2), (2, 1), (1, 2)]]
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

    def is_done(self) -> bool:
        """Current board's done status
        :return:
            bool: board's done status. True for the game is done, False otherwise
        """
        return self.board.result(claim_draw=True) != '*'

    def get_result(self) -> Any:
        """Get the game's result if done
        :return:
            Any: game's result. None for game is undone, str for game is done
        """
        if self.result is not None:
            return self.result
        else:
            if self.board.result(claim_draw=True) != '*':
                return self.board.result(claim_draw=True)
            else:
                return None

    def reset(self, fen: str = None, board2state: Board2State = None) -> Any:
        """Reset the game
        :param
            fen(str): the initial configuration for the board. None for default setting
        :return:
            str, Any: type of state parser, the board's current state
        """
        self.done = False
        self.result = None
        if fen is not None:
            self.board = chess.Board(fen=fen)
        else:
            self.board = chess.Board()
        _next_state = None
        if board2state is not None:
            _next_state = board2state.static_eval(self.board, self.actions)
        return board2state.__name__, _next_state

    def pass_move(self) -> None:
        """ Let the opponent play by not moving any piece
        :return:
        """
        self.board.turn = not self.board.turn

    def step(self, action: int, board2state: Board2State = None) -> Any:
        """Make an action in the environment
        :param
            action(int): choose the action
        :return:
            str: type of state parser,
            tuple(Any, float, bool, Any): next_state, reward, done, info. Following openai-gym's env
            if move is illegal then resign
            reward is 1 if win, -1 if lose. 0 if draw
        """
        _action = self.actions[action]
        _reward = 0
        _legal_move = chess.Move.from_uci(_action) in self.board.legal_moves
        _current_turn = self.board.turn
        _next_state = None
        if _legal_move:
            self.board.push_uci(_action)
            self.done = self.is_done()
            if self.done:
                if self.get_result() == '1-0':
                    if _current_turn:
                        _reward = 1
                    else:
                        _reward = -1
                elif self.get_result() == '0-1':
                    if _current_turn:
                        _reward = -1
                    else:
                        _reward = 1
                else:
                    _reward = 0
            if board2state is not None:
                _next_state = board2state.static_eval(self.board, self.actions)
                return board2state.__name__, _next_state, _reward, self.done, None
            else:
                return None, self.board.fen(), _reward, self.done, None
        else:
            return 'error', board2state.static_eval(self.board, self.actions), -1, self.done, None


if __name__ == '__main__':
    """For testing purpose only
    """
    print(len(ChessEnv.init_actions()))
    pass
