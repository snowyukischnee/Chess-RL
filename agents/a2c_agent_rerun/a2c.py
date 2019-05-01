from random import getrandbits
import numpy as np

import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../../')))
from CP_CHESS.agents.a2c_agent_rerun.model import Model
from CP_CHESS.agents.a2c_agent_rerun.envwrapper import ChessEnvWrapper
from CP_CHESS.agents.a2c_agent_rerun.memory import Memory
from CP_CHESS.env.environment import ChessEnv

actions = np.identity(len(ChessEnv.init_actions()))


class LR(object):
    def __init__(self):
        self.a_lr = None
        self.c_lr = None


class A2CConfig(object):
    memory_size = 1024
    max_steps = 1024
    update_interval = 128
    batch_size = 128
    a_lr = 0.1
    c_lr = 0.2
    decay_interval = 50
    decay_rate = 0.5
    model_update_interval = 10


class A2C(object):
    def __init__(self, gpu_idx: int = None, model_dir: str = None, model_ver: int = None):
        if model_dir is None:
            model_dir = './model'
        if model_ver is None:
            model_ver = 0
        self.model_dir = model_dir
        self.model_ver = model_ver
        # model
        self.model = Model(gpu_idx=gpu_idx)
        self.load_model(model_dir, model_ver)
        # env
        self.env = ChessEnvWrapper(self.model)
        self.memory = Memory(A2CConfig.memory_size)
        # learning rate
        self.lr = LR()
        self.lr.a_lr = A2CConfig.a_lr
        self.lr.c_lr = A2CConfig.c_lr

    def load_model(self, model_dir: str, model_ver: int) -> None:
        try:
            self.model.load('{}/model{}/model.ckpt'.format(model_dir, model_ver))
            print('model{} at {} was loaded'.format(model_dir, model_ver))
        except:
            print('model not found! save initial model instead')
            model_ver = 0
            self.model_ver = 0
            save_path = self.model.save('{}/model{}/model.ckpt'.format(model_dir, model_ver))
            print('initial model saved at {}'.format(save_path))

    def train(self, n_eps: int = 100):
        for episode in range(1, n_eps + 1):
            print('Episode {}'.format(episode))
            if episode % A2CConfig.decay_interval == 0:
                self.lr.a_lr *= A2CConfig.decay_rate
                self.lr.c_lr *= A2CConfig.decay_rate
            # ----------------------------------------------------------------------------------------------------------
            state_type, player_state = self.env.reset(player_white_pieces=not getrandbits(1))
            timestep = 0
            interval_clock = 0
            while True:
                timestep += 1
                interval_clock += 1
                player_action = self.model.act(player_state)
                print('state: {}, action: {}'.format(self.env.env.board.fen(), self.env.env.actions[player_action]))
                state_type, player_next_state, reward, done, info = self.env.step(player_action)
                print('next_state: {}, done: {}'.format(self.env.env.board.fen(), done))
                self.memory.add((player_state, actions[player_action], player_next_state, reward))
                if done or interval_clock > A2CConfig.update_interval or timestep > A2CConfig.max_steps:
                    interval_clock = 0
                    self.model.learn(self.memory.sample(A2CConfig.batch_size, continuous=True), a_lr=self.lr.a_lr, c_lr=self.lr.c_lr)
                if done or timestep > A2CConfig.max_steps:
                    break
                player_state = player_next_state
            # ----------------------------------------------------------------------------------------------------------
            if episode % A2CConfig.model_update_interval == 0:
                self.model_ver += 1
                save_path = self.model.save('{}/model{}/model.ckpt'.format(self.model_dir, self.model_ver))
                print('new model saved at {}'.format(save_path))


class PlayWBot(object):
    def __init__(self, model_dir: str = None, model_ver: int = None):
        if model_dir is None:
            model_dir = './model'
        if model_ver is None:
            model_ver = 0
        self.model_dir = model_dir
        self.model_ver = model_ver
        # model
        self.model = Model(gpu_idx=None)
        self.load_model(model_dir, model_ver)
        # env
        self.env = ChessEnvWrapper(self.model)

    def load_model(self, model_dir: str, model_ver: int) -> None:
        try:
            self.model.load('{}/model{}/model.ckpt'.format(model_dir, model_ver))
            print('model{} at {} was loaded'.format(model_dir, model_ver))
        except:
            print('model not found! save initial model instead')
            model_ver = 0
            self.model_ver = 0
            save_path = self.model.save('{}/model{}/model.ckpt'.format(model_dir, model_ver))
            print('initial model saved at {}'.format(save_path))

    def play(self, player_white_pieces: bool = True) -> None:
        done = False
        print(self.env.env.board)
        _, _ = self.env.reset(player_white_pieces=player_white_pieces)
        print(self.env.env.board)
        while True:
            action_str = input('Your move:')
            if action_str == 'exit':
                break
            elif action_str == 'pass':
                self.env.env.pass_move()
            else:
                action = self.env.env.actions.index(action_str)
                _, _, reward, done, _ = self.env.step(action)
            print(self.env.env.board)
            if done:
                print('Reward: {}'.format(reward))
                break