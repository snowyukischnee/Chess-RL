from typing import Any

import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../../')))
from CP_CHESS.agents.base_agent.agent import BaseAgent
# from CP_CHESS.env.environment import ChessEnv
# from CP_CHESS.agents.a2c_agent.board2state import Board2State0 as board2state
from CP_CHESS.agents.a2c_agent.config import Config
from CP_CHESS.agents.a2c_agent.model import Model


class Agent(BaseAgent):
    def __init__(self, config: Config, gpu_idx: int = None) -> None:
        super().__init__(config)
        self.model = Model(config, gpu_idx)
        self.model_dir = './model'
        self.model_ver = 0

    def action(self, state_type: str, state: Any, play: bool = False) -> int:
        _action = 0
        _action = self.model.act(state, play=play)
        return _action

    def save_model_replace(self) -> str:
        return self.model.save('{}/model{}/model.ckpt'.format(self.model_dir, self.model_ver))

    def save_model(self) -> str:
        self.model_ver += 1
        return self.model.save('{}/model{}/model.ckpt'.format(self.model_dir, self.model_ver))

    def load_model(self, model_dir: str, model_ver: int) -> None:
        try:
            self.model.load('{}/model{}/model.ckpt'.format(model_dir, model_ver))
            self.model_dir = model_dir
            self.model_ver = model_ver
        except:
            print('model not found! save initial model instead')
            self.model_ver = -1
            self.save_model()


if __name__ == '__main__':
    """For testing purpose only
    """
    # game = ChessEnv()
    # _, a = game.reset(fen=None, board2state=board2state)
    # tp, a, b, c, d = game.step(928, board2state=board2state)
    # config = Config()
    # config.n_action = len(game.actions)
    # ag = Agent(config)
    # x = ag.action(tp, a)