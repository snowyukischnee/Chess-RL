from typing import Any

import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../../')))
from CP_CHESS.agents.base_agent.agent import BaseAgent
from CP_CHESS.env.environment import ChessEnv
from CP_CHESS.utils.board2state import Board2State0 as board2state
from CP_CHESS.agents.my_agent.config import Config
from CP_CHESS.agents.my_agent.model import Model


class Agent(BaseAgent):
    def __init__(self, config: Config):
        self.model = Model(config)

    def action(self, state_type: str, state: Any, play: bool = False) -> int:
        _action = 0
        _action = self.model.act(state, play=play)
        print(state[0].shape, state[1].shape, state[2].shape, state[3].shape)
        return _action


if __name__ == '__main__':
    """For testing purpose only
    """
    game = ChessEnv()
    _, a = game.reset(fen=None, board2state=board2state)
    tp, a, b, c, d = game.step(928, board2state=board2state)
    config = Config()
    config.n_action = len(game.actions)
    ag = Agent(config)
    x = ag.action(tp, a)