from typing import Any
import chess

import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../')))


class Board2State(object):
    def __init__(self, board: chess.Board, actions: list) -> None:
        pass

    def eval(self) -> Any:
        pass

    @staticmethod
    def static_eval(board: chess.Board, actions: list) -> Any:
        pass