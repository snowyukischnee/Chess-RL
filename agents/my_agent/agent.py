from typing import Any
import numpy as np
import chess
import tensorflow as tf

import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../../')))
from CP_CHESS.env.environment import ChessEnv
from CP_CHESS.utils.board2state import Board2State0 as board2state


class Config(object):
    def __init__(self):
        self.GAMMA = 0.95
        self.a_lr = 1e-4
        self.c_lr = 2e-4
        self.n_action = None


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

    def feature_extraction(self, s_overview: Any, s_piece_pos: Any, s_atk_map: Any):
        # s_overview
        s_overview_l1 = tf.layers.Dense(32)(s_overview)
        s_overview_l2 = tf.layers.Dense(64)(s_overview_l1)
        # s_piece_pos
        s_piece_pos_f = tf.layers.Flatten()(s_piece_pos)
        s_piece_pos_l1 = tf.layers.Dense(256)(s_piece_pos_f)
        s_piece_pos_l2 = tf.layers.Dense(64)(s_piece_pos_l1)
        # s_atk_map
        s_atk_map_f = tf.layers.Flatten()(s_atk_map)
        s_atk_map_l1 = tf.layers.Dense(512)(s_atk_map_f)
        s_atk_map_l2 = tf.layers.Dense(128)(s_atk_map_l1)
        # merged
        merged = tf.concat([s_overview_l2, s_piece_pos_l2, s_atk_map_l2], axis=1)
        return merged

    def predict(self, state: Any):
        return 0

    def play(self, state: Any):
        return 0

    def save(self, path: str = './df_model/model.ckpt'):
        save_path = self.saver.save(self.sess, path)
        print('model saved at {}'.format(save_path))

    def load(self, path: str = './df_model/model.ckpt'):
        self.saver.restore(self.sess, path)


class Agent(object):
    def __init__(self, config: Config):
        self.model = Model(config)

    def action(self, state_type: str, state: Any, play: bool = False) -> int:
        _action = 0
        if play:
            _action = self.model.play(state)
        else:
            _action = self.model.predict(state)
        print(state[0].shape, state[1].shape, state[2].shape, state[3].shape)
        return _action


if __name__ == '__main__':
    """For testing purpose only
    """
    game = ChessEnv()
    _, a = game.reset(fen=None, board2state=board2state)
    tp, a, b, c, d = game.step(928, board2state=board2state)
    config = Config()
    config.n_action = len(game.actions)
    ag = Agent(config)
    ag.action(tp, a)