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
        # Feature extraction block
        self.a_s_merged = self.feature_extraction(self.state_overview, self.state_piece_positions, self.state_attack_map)
        self.s_merged_shape = 416
        self.a_ns_merged = tf.placeholder(tf.float32, [None, self.s_merged_shape], 'ns_merged')
        # Curiosity block
        self.curious_params, self.curious_out, self.int_reward = self.build_curiosity_net(self.a_s_merged, self.action, self.a_ns_merged, self.s_merged_shape, '', 'dyn_net')
        self.clipped_int_reward = tf.clip_by_value(self.int_reward, 0, 1)  # experiment
        self.cr_obj_f = tf.reduce_mean(self.int_reward)
        self.curiosity_optimizer = tf.train.RMSPropOptimizer(self.config.cs_lr).minimize(self.cr_obj_f, var_list=self.curious_params)
        # Actor block
        self.a_params, self.a_dist = self.build_actor_net(self.a_s_merged, self.state_legal_actions, self.config.n_action, '', 'pi')
        self.a_log_prob = tf.log(tf.reduce_sum(self.action * self.a_dist))
        self.a_obj_f = tf.reduce_mean(self.a_log_prob * self.advantage)
        self.actor_optimizer = tf.train.RMSPropOptimizer(self.config.a_lr).minimize(-self.a_obj_f)
        # Critic block
        self.c_s_merged = self.feature_extraction(self.state_overview, self.state_piece_positions, self.state_attack_map)
        self.c_params, self.c_v = self.build_critic_net(self.c_s_merged, '', 'critic')
        self.c_adv = self.total_reward + self.config.GAMMA * self.v_next - self.c_v
        self.c_obj_f = tf.reduce_mean(tf.square(self.c_adv))
        self.critic_optimizer = tf.train.RMSPropOptimizer(self.config.c_lr).minimize(self.cr_obj_f)
        # utility
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.initializers.global_variables())
        # test
        # ns_mg = self.sess.run(self.a_s_merged, feed_dict={
        #     self.state_overview: np.zeros((2, 17)),
        #     self.state_legal_actions: np.ones((2, self.config.n_action)),
        #     self.state_piece_positions: np.zeros((2, 12, 64)),
        #     self.state_attack_map: np.ones((2, 128, 64)),
        # })
        # print(self.sess.run(self.int_reward, feed_dict={
        #     self.state_overview: np.zeros((2, 17)),
        #     self.state_legal_actions: np.ones((2, self.config.n_action)),
        #     self.state_piece_positions: np.zeros((2, 12, 64)),
        #     self.state_attack_map: np.ones((2, 128, 64)),
        #     self.action: np.zeros((2, self.config.n_action)),
        #     self.a_ns_merged: ns_mg,
        # }))

    @staticmethod
    def build_curiosity_net(s_tensor: Any, a_tensor: Any, ns_tensor: Any, ns_shape: int, outer_scope: str, name: str, trainable: bool = True) -> Any:
        if outer_scope and outer_scope.strip():
            full_path = outer_scope + '/' + name
        else:
            full_path = name
        with tf.variable_scope(name):
            concat_tensor = tf.concat([s_tensor, tf.cast(a_tensor, dtype=tf.float32)], axis=1)
            l1 = tf.layers.Dense(
                units=512,
                activation=tf.nn.relu,
                trainable=trainable,
            )(concat_tensor)
            ns_predictor = tf.layers.Dense(
                units=ns_shape,
                activation=tf.nn.relu,
                trainable=trainable,
                name='curiosity_output'
            )(l1)
            int_reward = tf.reduce_sum(tf.square(ns_tensor - ns_predictor), axis=1)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=full_path)
        return params, ns_predictor, int_reward

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
            l1 = tf.layers.Dense(
                units=n_action,
                activation=tf.nn.relu,
                trainable=trainable,
            )(l1)
            l1_filtered = l1 * legal_action
            action_distribution = tf.nn.softmax(l1_filtered, axis=1, name='action_distribution')
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
        s_overview, s_legal_action, s_piece_pos, s_atk_map, action, ns_overview, ns_legal_action, ns_piece_pos, ns_atk_map, reward = zip(*experiences)
        s_overview = np.asarray(s_overview)
        s_legal_action = np.asarray(s_legal_action)
        s_piece_pos = np.asarray(s_piece_pos)
        s_atk_map = np.asarray(s_atk_map)
        action = np.asarray(action)
        ns_overview = np.asarray(ns_overview)
        ns_legal_action = np.asarray(ns_legal_action)
        ns_piece_pos = np.asarray(ns_piece_pos)
        ns_atk_map = np.asarray(ns_atk_map)
        reward = np.asarray(reward)
        v_next = self.sess.run(self.c_v, feed_dict={
            self.next_state_overview: ns_overview,
            self.next_state_legal_actions: ns_legal_action,
            self.next_state_piece_positions: ns_piece_pos,
            self.next_state_attack_map: ns_atk_map,
        })
        ns_merged = self.sess.run(self.a_s_merged, feed_dict={
            self.next_state_overview: ns_overview,
            self.next_state_legal_actions: ns_legal_action,
            self.next_state_piece_positions: ns_piece_pos,
            self.next_state_attack_map: ns_atk_map,
        })
        int_reward = self.sess.run(self.int_reward, feed_dict={
            self.state_overview: s_overview,
            self.state_legal_actions: s_legal_action,
            self.state_piece_positions: s_piece_pos,
            self.state_attack_map: s_atk_map,
            self.action: action,
            self.a_ns_merged: ns_merged
        })
        total_reward = np.add(reward, int_reward)
        advantage = self.sess.run(self.c_adv, feed_dict={
            self.state_overview: s_overview,
            self.state_legal_actions: s_legal_action,
            self.state_piece_positions: s_piece_pos,
            self.state_attack_map: s_atk_map,
            self.total_reward: total_reward,
            self.v_next: v_next
        })
        feed_dict = {
            self.state_overview: s_overview,
            self.state_legal_actions: s_legal_action,
            self.state_piece_positions: s_piece_pos,
            self.state_attack_map: s_atk_map,
            self.action: action,
            self.advantage: advantage,
            self.a_ns_merged: ns_merged
        }
        [self.sess.run(self.actor_optimizer, feed_dict=feed_dict) for _ in range(self.config.N_UPDATE)]
        [self.sess.run(self.critic_optimizer, feed_dict=feed_dict) for _ in range(self.config.N_UPDATE)]
        [self.sess.run(self.curiosity_optimizer, feed_dict=feed_dict) for _ in range(self.config.N_UPDATE)]

    def act(self, state: Any, play: bool = False) -> int:
        s_overview, s_legal_action, s_piece_pos, s_atk_map = state  # not zip
        action = self.sess.run(self.a_dist, feed_dict={
            self.state_overview: np.expand_dims(s_overview, axis=0),
            self.state_legal_actions: np.expand_dims(s_legal_action, axis=0),
            self.state_piece_positions: np.expand_dims(s_piece_pos, axis=0),
            self.state_attack_map: np.expand_dims(s_atk_map, axis=0),
        })
        action = np.squeeze(action, axis=0)
        if play is False:
            return np.random.choice(action.shape[0], 1, p=action)
        else:
            return np.argmax(action)

    def save(self, path: str = './df_model/model.ckpt') -> None:
        save_path = self.saver.save(self.sess, path)
        print('model saved at {}'.format(save_path))

    def load(self, path: str = './df_model/model.ckpt') -> None:
        self.saver.restore(self.sess, path)