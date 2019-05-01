from typing import Any
import numpy as np
import tensorflow as tf
from random import getrandbits

import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../../')))
from CP_CHESS.agents.a2c_agent_exp.envwrapper import ChessEnvWrapper
from CP_CHESS.agents.a2c_agent_exp.model import Model
from CP_CHESS.agents.a2c_agent_exp.memory import Memory
from CP_CHESS.env.environment import ChessEnv

actions = np.identity(len(ChessEnv.init_actions()))


class LR(object):
    def __init__(self):
        self.a_lr = None
        self.c_lr = None


class TrainConfig(object):
    memory_size = 1024
    max_steps = 1024


class Worker(object):
    def __init__(self, graph: tf.Graph, sess: tf.Session, gpu_idx: int = None, name: str = 'Local_Net', global_net: Any = None):
        self.env = ChessEnvWrapper(name)
        self.name = name
        self.model = Model(graph, sess, gpu_idx=gpu_idx, scope=name, global_net=global_net)
        self.memory = Memory(TrainConfig.memory_size)
        self.update_feed_dict = None

    def learn(self, experiences: Any, lr: LR) -> None:
        self.update_feed_dict = None
        state, action, next_state, reward = zip(*experiences)
        s_overview, s_legal_action, s_piece_pos, s_atk_map = zip(*state)
        ns_overview, ns_legal_action, ns_piece_pos, ns_atk_map = zip(*next_state)
        s_overview = np.asarray(s_overview)
        s_legal_action = np.asarray(s_legal_action)
        s_piece_pos = np.asarray(s_piece_pos)
        s_atk_map = np.asarray(s_atk_map)
        action = np.asarray(action)
        ns_overview = np.asarray(ns_overview)
        ns_legal_action = np.asarray(ns_legal_action)
        ns_piece_pos = np.asarray(ns_piece_pos)
        ns_atk_map = np.asarray(ns_atk_map)
        reward = np.asarray(reward)[:, np.newaxis]
        v_next = self.model.sess.run(self.model.c_v, feed_dict={
            self.model.s_overview: ns_overview,
            self.model.s_legal_actions: ns_legal_action,
            self.model.s_piece_positions: ns_piece_pos,
            self.model.s_attack_map: ns_atk_map,
        })
        advantage = self.model.sess.run(self.model.c_adv, feed_dict={
            self.model.s_overview: s_overview,
            self.model.s_legal_actions: s_legal_action,
            self.model.s_piece_positions: s_piece_pos,
            self.model.s_attack_map: s_atk_map,
            self.model.reward: reward,
            self.model.v_next: v_next
        })
        feed_dict = {
            self.model.s_overview: s_overview,
            self.model.s_legal_actions: s_legal_action,
            self.model.s_piece_positions: s_piece_pos,
            self.model.s_attack_map: s_atk_map,
            self.model.action: action,
            self.model.advantage: advantage,
            self.model.v_next: v_next,
            self.model.reward: reward,
            self.model.a_lr: lr.a_lr,
            self.model.c_lr: lr.c_lr,
        }
        self.update_feed_dict = feed_dict

    def work(self, coord: tf.train.Coordinator, lr: LR):
        self.model.pull_global()
        # self.env.load_model(TrainConfig.model_dir, TrainConfig.model_ver)
        state_type, player_state = self.env.reset(player_white_pieces=not getrandbits(1))
        timestep = 0
        while True:
            if coord.should_stop():
                break
            timestep += 1
            player_action = self.model.act(player_state)
            # print('Worker: {} state: {}, action: {}'.format(self.name, self.env.env.board.fen(), self.env.env.actions[player_action]))
            state_type, player_next_state, reward, done, info = self.env.step(player_action)
            # print('Worker: {} next_state: {}, done: {}'.format(self.name, self.env.env.board.fen(), done))
            self.memory.add((player_state, actions[player_action], player_next_state, reward))
            if done or timestep > TrainConfig.max_steps:
                self.learn(self.memory.sample(TrainConfig.max_steps, continuous=True), lr)
                break
            player_state = player_next_state
