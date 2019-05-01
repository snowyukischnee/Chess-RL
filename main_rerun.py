import argparse
from tensorflow.python.client import device_lib


import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../')))
from CP_CHESS.agents.a2c_agent_rerun.a2c import A2C, PlayWBot
from CP_CHESS.utils.playwengine import PlayWEngine

def get_available_devices():
    local_devices = device_lib.list_local_devices()
    return [device.name for device in local_devices if device.device_type == 'GPU']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility for CP_CHESS')
    parser_group = parser.add_mutually_exclusive_group()
    parser_group.add_argument('--train', action='store_true', default=False, help='bot self-play')
    parser_group.add_argument('--play', action='store_true', default=False, help='play with bot (console)')
    parser.add_argument('-r', '--resume', action='store', dest='model_ver', type=int, default=None, help='model version to be resumed')
    parser.add_argument('-e', '--episode', action='store', dest='n_eps', type=int, default=100, help='model version to be resumed')
    parser.add_argument('--gpu', action='store', type=int, default=None, help='gpu for training process')
    parser.add_argument('--version', action='store', default=0, help='model version to be loaded')
    parser.add_argument('--black', action='store_true', default=False, help='player piece color (B|W)')
    args = parser.parse_args()
    if args.train:
        gpu_idx = args.gpu
        if gpu_idx is not None:
            gpus = get_available_devices()
            if gpu_idx < 0 or gpu_idx >= len(gpus):
                print('GPU index is invalid')
                exit()
        a2c = A2C(gpu_idx=gpu_idx, model_dir='./model', model_ver=args.model_ver)
        a2c.train(args.n_eps)
    elif args.play:
        pwb = PlayWBot(model_dir='./model', model_ver=args.version)
        pwb.play(player_white_pieces=not args.black)
    else:
        print('Play with stockfish')
        pwe = PlayWEngine(bin_path='./stockfish/stockfish_win10_64', timelimit=0.1)
        pwe.play(player_white_pieces=not args.black)