from typing import Any
import numpy as np
import chess
import tensorflow as tf

import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../../')))
from CP_CHESS.agents.my_agent.config import Config


class Model(object):
    def __init__(self, config: Config):
        self.config = config
        # state
        self.state_overview = tf.placeholder(tf.float32, [None, 17], 'state_overview')
        self.state_legal_actions = tf.placeholder(tf.float32, [None, self.config.n_action], 'state_legal_actions')
        self.state_piece_positions = tf.placeholder(tf.float32, [None, 12, 64], 'state_piece_positions')
        self.state_attack_map = tf.placeholder(tf.float32, [None, 128, 64], 'state_attack_map')
        # action
        self.action = tf.placeholder(tf.float32, [None, self.config.n_action], 'action')
        # next_state
        self.next_state_overview = tf.placeholder(tf.float32, [None, 17], 'next_state_overview')
        self.next_state_legal_actions = tf.placeholder(tf.float32, [None, self.config.n_action], 'next_state_legal_actions')
        self.next_state_piece_positions = tf.placeholder(tf.float32, [None, 12, 64], 'next_state_piece_positions')
        self.next_state_attack_map = tf.placeholder(tf.float32, [None, 128, 64], 'next_state_attack_map')
        # external reward
        self.reward = tf.placeholder(tf.float32, [None, 1], 'reward_ext')

        self.v_next = tf.placeholder(tf.float32, [None, 1], 'v_next')
        self.advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.total_reward = tf.placeholder(tf.float32, [None, 1], 'total_reward')

        self.s_merged = self.feature_extraction(self.state_overview, self.state_piece_positions, self.state_attack_map)
        self.ns_merged = self.feature_extraction(self.next_state_overview, self.next_state_piece_positions, self.next_state_attack_map)
        self.s_merged_shape = 416
        # Curiosity block
        self.curious_params, self.curious_out, self.int_reward = self.build_curiosity_net(self.ns_merged, self.s_merged_shape, '', 'Curiosity')
        self.curiosity_optimizer = tf.train.RMSPropOptimizer(self.config.cs_lr).minimize(tf.reduce_mean(self.int_reward), var_list=self.curious_params)
        # Actor block
        with tf.variable_scope('Actor'):
            pass
        # Critic block
        with tf.variable_scope('Critic'):
            pass

        with tf.Session() as sess:
            sess.run(tf.initializers.global_variables())
            print(
                sess.run(
                    [tf.shape(self.s_merged)],
                    feed_dict={
                        self.state_overview: np.ones((10, 17), dtype=np.int32),
                        self.state_piece_positions: np.ones((10, 12, 64), dtype=np.int32),
                        self.state_attack_map: np.ones((10, 128, 64), dtype=np.int32)
                    }
                )
            )

        self.saver = tf.train.Saver()
        self.sess = tf.Session()

    @staticmethod
    def build_curiosity_net(input_tensor: Any, output_shape: int, outer_scope: str, name: str, trainable: bool = True) -> Any:
        if outer_scope and outer_scope.strip():
            full_path = outer_scope + '/' + name
        else:
            full_path = name
        rand_encode_ns = tf.layers.Dense(units=512, activation=tf.nn.relu, trainable=trainable)(input_tensor)
        rand_encode_ns_predictor = tf.layers.Dense(units=output_shape, activation=tf.nn.relu, trainable=trainable)(rand_encode_ns)
        int_reward = tf.reduce_sum(tf.square(input_tensor - rand_encode_ns_predictor, trainable=trainable), axis=1)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=full_path)
        return params, rand_encode_ns_predictor, int_reward

    @staticmethod
    def build_actor_net(self, input_tensor: Any, outer_scope: str, name: str, trainable: bool = True) -> Any:
        pass

    @staticmethod
    def build_critic_net(self, input_tensor: Any, outer_scope: str, name: str, trainable: bool = True) -> Any:
        pass

    @staticmethod
    def update_network_op(origin: Any, target: Any, tau: float = None) -> Any:
        if 0 <= tau <= 1:
            return [t.assign(tau * o + (1 - tau) * t) for o, t in zip(origin, target)]
        else:
            return [t.assign(o) for o, t in zip(origin, target)]

    @staticmethod
    def feature_extraction(s_overview: Any, s_piece_pos: Any, s_atk_map: Any):
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

    def predict(self, state: Any):
        return 0

    def play(self, state: Any):
        return 0

    def save(self, path: str = './df_model/model.ckpt'):
        save_path = self.saver.save(self.sess, path)
        print('model saved at {}'.format(save_path))

    def load(self, path: str = './df_model/model.ckpt'):
        self.saver.restore(self.sess, path)