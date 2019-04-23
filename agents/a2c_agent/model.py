from typing import Any
import numpy as np
import chess
import tensorflow as tf

import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../../')))
from CP_CHESS.agents.a2c_agent.config import Config


class Model(object):
    def __init__(self, config: Config, gpu_idx: int = None) -> None:
        self.config = config
        if gpu_idx is None:
            device_name = '/cpu:0'
        else:
            device_name = '/gpu:{}'.format(gpu_idx)
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device(device_name):
                self.state_overview = tf.placeholder(tf.float32, [None, 17], 'state_overview')
                self.state_legal_actions = tf.placeholder(tf.float32, [None, self.config.n_action], 'state_legal_actions')
                self.state_piece_positions = tf.placeholder(tf.float32, [None, 12, 64], 'state_piece_positions')
                self.state_attack_map = tf.placeholder(tf.float32, [None, 128, 64], 'state_attack_map')
                # action
                self.action = tf.placeholder(tf.float32, [None, self.config.n_action], 'action')

                self.v_next = tf.placeholder(tf.float32, [None, 1], 'v_next')
                self.advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')
                self.reward = tf.placeholder(tf.float32, [None, 1], 'reward')
                # Feature extraction block
                self.a_s_merged = self.feature_extraction(self.state_overview, self.state_piece_positions, self.state_attack_map)
                self.s_merged_shape = 416
                self.a_ns_merged = tf.placeholder(tf.float32, [None, self.s_merged_shape], 'ns_merged')
                # Actor block
                self.a_params, self.a_dist = self.build_actor_net(self.a_s_merged, self.state_legal_actions, self.config.n_action, '', 'pi')
                self.a_log_prob = tf.log(tf.reduce_sum(self.action * self.a_dist))
                self.a_obj_f = tf.reduce_mean(self.a_log_prob * self.advantage)
                self.actor_optimizer = tf.train.AdamOptimizer(self.config.a_lr).minimize(-self.a_obj_f)
                # Critic block
                self.c_s_merged = self.feature_extraction(self.state_overview, self.state_piece_positions, self.state_attack_map)
                self.c_params, self.c_v = self.build_critic_net(self.c_s_merged, '', 'critic')
                self.c_adv = self.reward + self.config.GAMMA * self.v_next - self.c_v
                self.c_obj_f = tf.reduce_mean(tf.square(self.c_adv))
                self.critic_optimizer = tf.train.AdamOptimizer(self.config.c_lr).minimize(self.c_obj_f)
                # utility
                self.sess = tf.Session()
            with tf.device('/cpu:0'):
                self.sess.run(tf.initializers.global_variables())
                self.saver = tf.train.Saver()

    @staticmethod
    def build_actor_net(input_tensor: Any, legal_action: Any, n_action: int, outer_scope: str, name: str, trainable: bool = True) -> Any:
        if outer_scope and outer_scope.strip():
            full_path = outer_scope + '/' + name
        else:
            full_path = name
        with tf.variable_scope(name):
            l1 = tf.layers.Dense(
                units=512,
                activation=tf.nn.relu,
                trainable=trainable,
            )(input_tensor)
            action_distribution = tf.layers.Dense(
                units=n_action,
                activation=tf.nn.softmax,
                trainable=trainable,
            )(l1)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=full_path)
        return params, action_distribution

    @staticmethod
    def build_critic_net(input_tensor: Any, outer_scope: str, name: str, trainable: bool = True) -> Any:
        if outer_scope and outer_scope.strip():
            full_path = outer_scope + '/' + name
        else:
            full_path = name
        with tf.variable_scope(name):
            l1 = tf.layers.Dense(
                units=512,
                activation=tf.nn.relu,
                trainable=trainable,
            )(input_tensor)
            v = tf.layers.Dense(
                units=1,
                activation=tf.nn.relu,
                trainable=trainable,
            )(l1)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=full_path)
        return params, v

    @staticmethod
    def update_network_op(origin: Any, target: Any, tau: float = None) -> Any:
        if 0 <= tau <= 1:
            return [t.assign(tau * o + (1 - tau) * t) for o, t in zip(origin, target)]
        else:
            return [t.assign(o) for o, t in zip(origin, target)]

    @staticmethod
    def feature_extraction(s_overview: Any, s_piece_pos: Any, s_atk_map: Any) -> Any:
        # s_overview
        s_overview_l1 = tf.layers.Dense(units=32, activation=tf.nn.relu)(s_overview)
        # s_piece_pos
        s_piece_pos_f = tf.layers.Flatten()(s_piece_pos)
        s_piece_pos_l1 = tf.layers.Dense(units=128, activation=tf.nn.relu)(s_piece_pos_f)
        # s_atk_map
        s_atk_map_f = tf.layers.Flatten()(s_atk_map)
        s_atk_map_l1 = tf.layers.Dense(units=256, activation=tf.nn.relu)(s_atk_map_f)
        # merged
        merged = tf.concat([s_overview_l1, s_piece_pos_l1, s_atk_map_l1], axis=1)
        return merged  # shape [None, 416]

    def learn(self, experiences: Any) -> None:
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
        v_next = self.sess.run(self.c_v, feed_dict={
            self.state_overview: ns_overview,
            self.state_legal_actions: ns_legal_action,
            self.state_piece_positions: ns_piece_pos,
            self.state_attack_map: ns_atk_map,
        })
        ns_merged = self.sess.run(self.a_s_merged, feed_dict={
            self.state_overview: ns_overview,
            self.state_legal_actions: ns_legal_action,
            self.state_piece_positions: ns_piece_pos,
            self.state_attack_map: ns_atk_map,
        })
        advantage = self.sess.run(self.c_adv, feed_dict={
            self.state_overview: s_overview,
            self.state_legal_actions: s_legal_action,
            self.state_piece_positions: s_piece_pos,
            self.state_attack_map: s_atk_map,
            self.reward: reward,
            self.v_next: v_next
        })
        feed_dict = {
            self.state_overview: s_overview,
            self.state_legal_actions: s_legal_action,
            self.state_piece_positions: s_piece_pos,
            self.state_attack_map: s_atk_map,
            self.action: action,
            self.advantage: advantage,
            self.a_ns_merged: ns_merged,
            self.v_next: v_next,
            self.reward: reward,
        }
        [self.sess.run(self.actor_optimizer, feed_dict=feed_dict) for _ in range(self.config.N_UPDATE)]
        [self.sess.run(self.critic_optimizer, feed_dict=feed_dict) for _ in range(self.config.N_UPDATE)]

    def act(self, state: Any, play: bool = False) -> int:
        s_overview, s_legal_action, s_piece_pos, s_atk_map = state  # not zip
        action = self.sess.run(self.a_dist, feed_dict={
            self.state_overview: np.expand_dims(s_overview, axis=0),
            self.state_legal_actions: np.expand_dims(s_legal_action, axis=0),
            self.state_piece_positions: np.expand_dims(s_piece_pos, axis=0),
            self.state_attack_map: np.expand_dims(s_atk_map, axis=0),
        })
        action = np.squeeze(action, axis=0)
        action[s_legal_action == 0] = np.NINF
        # debug
        # print('X = ', action)
        # print('X = ', action[action != np.NINF])
        action = action - action.max(axis=0, keepdims=True)
        action = np.exp(action)
        action = action / action.sum(axis=0, keepdims=True)
        if play is False:
            # from CP_CHESS.env.environment import ChessEnv
            # actions = ChessEnv.init_actions()
            # c = [s_legal_action == 1]
            # print(np.array(actions)[c])
            # print(s_overview[0] == chess.WHITE)
            # import matplotlib.pyplot as plt
            # plt.plot(action)
            # DIR = './image'
            # index = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
            # plt.savefig('./image/a{}.png'.format(index))
            # plt.clf()
            return int(np.random.choice(action.shape[0], 1, p=action)[0])
        else:
            return int(np.argmax(action))

    def save(self, path: str = './df_model/model.ckpt') -> str:
        save_path = self.saver.save(self.sess, path)
        return save_path

    def load(self, path: str = './df_model/model.ckpt') -> None:
        self.saver.restore(self.sess, path)