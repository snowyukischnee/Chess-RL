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

        s_merged = self.feature_extraction(self.state_overview, self.state_piece_positions, self.state_attack_map)
        ns_merged = self.feature_extraction(self.next_state_overview, self.next_state_piece_positions, self.next_state_attack_map)

        with tf.Session() as sess:
            sess.run(tf.initializers.global_variables())
            print(
                sess.run(
                    [tf.shape(s_merged)],
                    feed_dict={
                        self.state_overview: np.ones((10, 17), dtype=np.int32),
                        self.state_piece_positions: np.ones((10, 12, 64), dtype=np.int32),
                        self.state_attack_map: np.ones((10, 128, 64), dtype=np.int32)
                    }
                )
            )

        self.saver = tf.train.Saver()
        self.sess = tf.Session()

    def update_network(self, origin: Any, target: Any, tau: float = None) -> None:
        if 0 <= tau <= 1:
            self.sess.run([t.assign(tau * o + (1 - tau) * t) for o, t in zip(origin, target)])
        else:
            self.sess.run([t.assign(o) for o, t in zip(origin, target)])

    def feature_extraction(self, s_overview: Any, s_piece_pos: Any, s_atk_map: Any):
        # s_overview
        s_overview_l1 = tf.layers.Dense(32)(s_overview)
        # s_piece_pos
        s_piece_pos_f = tf.layers.Flatten()(s_piece_pos)
        s_piece_pos_l1 = tf.layers.Dense(128)(s_piece_pos_f)
        # s_atk_map
        s_atk_map_f = tf.layers.Flatten()(s_atk_map)
        s_atk_map_l1 = tf.layers.Dense(256)(s_atk_map_f)
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