import argparse
from random import getrandbits

import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../')))
from CP_CHESS.agents.a2c_agent.config import Config
from CP_CHESS.agents.a2c_agent.agent import Agent
from CP_CHESS.play.selfplay import SelfPlayConfig, SelfPlay
from CP_CHESS.play.playwbot import PlayConfig, PlayWBot
from tensorflow.python.client import device_lib


def get_available_devices():
    local_devices = device_lib.list_local_devices()
    return [device.name for device in local_devices if device.device_type == 'GPU']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility for CP_CHESS')
    parser_group = parser.add_mutually_exclusive_group()
    parser_group.add_argument('-t', '--train', action='store_true', default=False, help='bot self-play')
    parser_group.add_argument('-p', '--play', action='store_true', default=False, help='play with bot (console)')
    parser.add_argument('-r', '--resume', action='store', dest='model_version', type=int, default=None, help='model version to be resumed')
    parser.add_argument('-e', '--episode', action='store', dest='n_episodes', type=int, default=100, help='model version to be resumed')
    parser.add_argument('-g', '--gpu', action='store', type=int, default=None, help='gpu for training process')
    parser.add_argument('-v', '--version', action='store', default=0, help='model version to be loaded')
    parser.add_argument('-b', '--black', action='store_true', default=False, help='player piece color (B|W)')
    args = parser.parse_args()
    if args.train:
        gpu_idx = args.gpu
        if gpu_idx is not None:
            gpus = get_available_devices()
            if gpu_idx < 0 or gpu_idx >= len(gpus):
                print('GPU index is invalid')
                exit()
        sp_config = SelfPlayConfig()
        sp = SelfPlay(sp_config)
        sp.init(Config(), Agent, gpu_idx)
        if args.model_version is not None:
            sp.resume(sp.config.model_dir, args.model_version)
            print('model version {} will be loaded'.format(args.model_version))
        for v in range(args.n_episodes):
            fl = not getrandbits(1)
            sp.process(opponent_is_white=False)
    elif args.play:
        playwbot = PlayWBot(PlayConfig())
        playwbot.init(Config(), Agent)
        playwbot.load_model('./model', args.version)
        playwbot.play(player_is_white=not args.black)
    else:
        print('No parameters supplied. Do nothing!')