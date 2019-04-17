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
        self.n_episodes = 100
        self.model_dir = './model'
        self.model_ver = 0


class SelfPlay(object):
    def __init__(self, config: SelfPlayConfig, agent: BaseAgent):
        self.config = config
        self.current_player = None
        self.target_player = None
        self.env = None

    def init(self, config: Config):
        self.env = ChessEnv()
        self.current_player = Agent(config)
        self.target_player = Agent(config)

    def process(self, opponent_is_white: bool = False):
        self.target_player.load_model(self.config.model_dir, self.config.model_ver)
        if opponent_is_white is True:
            for episode in range(self.config.n_episodes):
                pass
        else:
            for episode in range(self.config.n_episodes):
                pass
        self.current_player.save_model()
        self.config.model_ver += 1


if __name__ == '__main__':
    env = ChessEnv()
    config = Config()
    config.n_action = len(env.actions)
    _, a = env.reset(fen=None, board2state=board2state)
    tp, a, b, c, d = env.step(928, board2state=board2state)
    ag = Agent(config)
    x = ag.action(tp, a)
