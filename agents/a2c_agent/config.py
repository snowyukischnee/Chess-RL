import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../../')))
from CP_CHESS.agents.a2c_agent.base_config import BaseConfig


class Config(BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.N_UPDATE = 10
        self.GAMMA = 0.95
        self.a_lr = 1e-4
        self.c_lr = 2e-4
        self.n_action = None
