import argparse
from random import getrandbits

import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../')))

from CP_CHESS.agents.my_agent.config import Config
from CP_CHESS.agents.my_agent.agent import Agent
from CP_CHESS.play.selfplay import SelfPlayConfig, SelfPlay
from CP_CHESS.play.playwbot import PlayConfig, PlayWBot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility for CP_CHESS')
    parser_group = parser.add_mutually_exclusive_group()
    parser_group.add_argument('-t', '--train', action='store_true', default=False, help='bot self-play')
    parser_group.add_argument('-p', '--play', action='store_true', default=False, help='play with bot (console)')
    parser.add_argument('-r', '--resume', action='store', dest='model_version', type=int, default=None, help='model version to be resumed')
    parser.add_argument('-e', '--episode', action='store', dest='n_episodes', type=int, default=100, help='model version to be resumed')
    parser.add_argument('-b', '--black', action='store_true', default=False, help='player piece color (B|W)')
    args = parser.parse_args()
    if args.train:
        print(args.n_episodes, args.model_version)
        sp_config = SelfPlayConfig()
        sp = SelfPlay(sp_config)
        sp.init(Config(), Agent)
        if args.model_version is not None:
            sp.resume(sp.config.model_dir, args.model_version)
            print('model version {} will be loaded'.format(args.model_version))
        for v in range(args.n_episodes):
            fl = not getrandbits(1)
            sp.process(opponent_is_white=fl)
    elif args.play:
        playwbot = PlayWBot(PlayConfig())
        playwbot.init(Config(), Agent)
        playwbot.load_model('./model', 60)
        playwbot.play(player_is_white=not args.black)
    else:
        print('No parameters supplied. Do nothing!')