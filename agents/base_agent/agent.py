from typing import Any

import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../../')))
from CP_CHESS.agents.base_agent.config import BaseConfig


class BaseAgent(object):
    def __init__(self, config: BaseConfig) -> None:
        pass

    def action(self, state_type: str, state: Any, play: bool = False) -> Any:
        pass

    def save_model(self):
        pass

    def load_model(self, model_dir: str, model_ver: int):
        pass
