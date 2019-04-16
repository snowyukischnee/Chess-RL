from typing import Any

import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../')))

from CP_CHESS.agents.base_agent.agent import BaseAgent
from CP_CHESS.agents.base_agent.config import BaseConfig

from CP_CHESS.env.environment import ChessEnv
from CP_CHESS.utils.board2state import Board2State0 as board2state
from CP_CHESS.agents.my_agent.config import Config
from CP_CHESS.agents.my_agent.agent import Agent


class SelfPlayConfig(object):
    def __init__(self):
        self.n_episodes = 10


class SelfPlay(object):
    def __init__(self, config: SelfPlayConfig, agent: BaseAgent):
        pass


if __name__ == '__main__':
    env = ChessEnv()
    config = Config()
    config.n_action = len(env.actions)
    _, a = env.reset(fen=None, board2state=board2state)
    tp, a, b, c, d = env.step(928, board2state=board2state)
    ag = Agent(config)
    x = ag.action(tp, a)
