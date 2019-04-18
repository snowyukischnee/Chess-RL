from random import getrandbits

import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../')))

from CP_CHESS.agents.my_agent.config import Config
from CP_CHESS.agents.my_agent.agent import Agent
from CP_CHESS.play.selfplay import SelfPlayConfig
from CP_CHESS.play.selfplay import SelfPlay
from CP_CHESS.env.environment import ChessEnv

if __name__ == '__main__':
    sp_config = SelfPlayConfig()
    sp = SelfPlay(sp_config)
    sp.init(Config(), Agent)
    for v in range(100):
        fl = not getrandbits(1)
        sp.process(opponent_is_white=fl)
