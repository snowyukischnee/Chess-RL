from typing import Any
import numpy as np
import chess
import tensorflow as tf

import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../../')))
from CP_CHESS.env.environment import ChessEnv
# for debugging
np.set_printoptions(threshold=sys.maxsize)


class ModelConfig(object):
    N_UPDATE = 10
    GAMMA = 0.95
    a_lr = 1e-4
    c_lr = 2e-4
    n_action = 1968 # number of possible actions


class Model(object):
    GLOBAL_SCOPE = 'Global_Net'

    def __init__(self, graph: tf.Graph, sess: tf.Session, gpu_idx: int = None, scope: str = None, global_net: Any = None):
        if gpu_idx is None:
            device_name = '/cpu:0'
        else:
            device_name = '/gpu:{}'.format(gpu_idx)
        if Model.GLOBAL_SCOPE in scope:
            self.sess = None  # initialize later
            self.graph = tf.Graph()
        else:
            self.sess = sess
            self.graph = graph
        with self.graph.as_default():
            if Model.GLOBAL_SCOPE in scope:
                with tf.device(device_name):
                    with tf.variable_scope(scope):
                        # state
                        self.s_overview = tf.placeholder(tf.float32, [None, 17], 's_overview')
                        print('GLOBAL', scope, self.s_overview.graph)
                        self.s_legal_actions = tf.placeholder(tf.float32, [None, ModelConfig.n_action], 's_legal_actions')
                        self.s_piece_positions = tf.placeholder(tf.float32, [None, 12, 64], 's_piece_positions')
                        self.s_attack_map = tf.placeholder(tf.float32, [None, 128, 64], 's_attack_map')
                        # action
                        self.action = tf.placeholder(tf.float32, [None, ModelConfig.n_action], 'action')
                        # reward
                        self.reward = tf.placeholder(tf.float32, [None, 1], 'reward')
                        # values
                        self.v_next = tf.placeholder(tf.float32, [None, 1], 'v_next')
                        self.advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')
                        # next_state
                        # actor
                        self.a_params, self.a_dist = Model.build_actor_net(self.s_overview, self.s_piece_positions, self.s_attack_map, self.s_legal_actions, ModelConfig.n_action, scope, 'pi')
                        # critic
                        self.c_params, self.c_v = Model.build_critic_net(self.s_overview, self.s_piece_positions, self.s_attack_map, scope, 'critic')
                with tf.device('/cpu:0'):
                    self.saver = tf.train.Saver(var_list={v.op.name: v for v in [*self.a_params, *self.c_params]})
                self.sess = tf.Session(graph=self.graph)
                self.init_op = tf.global_variables_initializer()
            else:
                with tf.device(device_name):
                    with tf.variable_scope(scope):
                        # learning rate
                        self.a_lr = tf.placeholder(tf.float32, [], 'actor_lr')
                        self.c_lr = tf.placeholder(tf.float32, [], 'critic_lr')
                        # state
                        self.s_overview = tf.placeholder(tf.float32, [None, 17], 's_overview')
                        print('LOCAL', scope, self.s_overview.graph)
                        self.s_legal_actions = tf.placeholder(tf.float32, [None, ModelConfig.n_action], 's_legal_actions')
                        self.s_piece_positions = tf.placeholder(tf.float32, [None, 12, 64], 's_piece_positions')
                        self.s_attack_map = tf.placeholder(tf.float32, [None, 128, 64], 's_attack_map')
                        # action
                        self.action = tf.placeholder(tf.float32, [None, ModelConfig.n_action], 'action')
                        # reward
                        self.reward = tf.placeholder(tf.float32, [None, 1], 'reward')
                        # values
                        self.v_next = tf.placeholder(tf.float32, [None, 1], 'v_next')
                        self.advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')
                        # next_state
                        # actor
                        self.a_params, self.a_dist = Model.build_actor_net(self.s_overview, self.s_piece_positions, self.s_attack_map, self.s_legal_actions, ModelConfig.n_action, scope, 'pi')
                        self.a_log_prob = tf.log(tf.reduce_sum(self.action * self.a_dist))
                        self.a_obj_f = tf.reduce_mean(self.a_log_prob * self.advantage)
                        # critic
                        self.c_params, self.c_v = Model.build_critic_net(self.s_overview, self.s_piece_positions, self.s_attack_map, scope, 'critic')
                        self.c_adv = self.reward + ModelConfig.GAMMA * self.v_next - self.c_v
                        self.c_obj_f = tf.reduce_mean(tf.square(self.c_adv))
                        # optimizer
                        self.actor_optimizer = tf.train.AdamOptimizer(self.a_lr)
                        self.critic_optimizer = tf.train.AdamOptimizer(self.c_lr)
                        with tf.name_scope('local_grad'):
                            self.a_grads = tf.gradients(-self.a_obj_f, self.a_params)
                            self.c_grads = tf.gradients(self.c_obj_f, self.c_params)
                    with tf.name_scope('sync'):
                        with tf.name_scope('pull'):
                            self.pull_a_params = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, global_net.a_params)]
                            self.pull_c_params = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, global_net.c_params)]
                        with tf.name_scope('push'):
                            self.push_a_params = self.actor_optimizer.apply_gradients(zip(self.a_grads, global_net.a_params))
                            self.push_c_params = self.critic_optimizer.apply_gradients(zip(self.c_grads, global_net.c_params))

    @staticmethod
    def feature_extraction(s_overview: Any, s_piece_pos: Any, s_atk_map: Any, trainable: bool = True) -> Any:
        s_overview_l1 = tf.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.constant_initializer(0.1),
            trainable=trainable
        )(s_overview)
        s_piece_pos_f = tf.layers.Flatten()(s_piece_pos)
        s_piece_pos_l1 = tf.layers.Dense(
            units=128,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.constant_initializer(0.1),
            trainable=trainable
        )(s_piece_pos_f)
        s_atk_map_f = tf.layers.Flatten()(s_atk_map)
        s_atk_map_l1 = tf.layers.Dense(
            units=256,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.constant_initializer(0.1),
            trainable=trainable
        )(s_atk_map_f)
        merged = tf.concat([s_overview_l1, s_piece_pos_l1, s_atk_map_l1], axis=1)
        return merged  # shape [None, 416]

    @staticmethod
    def build_actor_net(s_overview: Any, s_piece_pos: Any, s_atk_map: Any, legal_action: Any, n_action: int, outer_scope: str, name: str, trainable: bool = True) -> Any:
        if outer_scope and outer_scope.strip():
            full_path = outer_scope + '/' + name
        else:
            full_path = name
        with tf.variable_scope(name):
            input_tensor = Model.feature_extraction(s_overview, s_piece_pos, s_atk_map, trainable)
            l1 = tf.layers.Dense(
                units=512,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.constant_initializer(0.1),
                trainable=trainable,
            )(input_tensor)
            action_distribution = tf.layers.Dense(
                units=n_action,
                activation=tf.nn.tanh,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.constant_initializer(0.1),
                trainable=trainable,
            )(l1)
            action_distribution = tf.math.softplus(action_distribution)
            action_distribution = action_distribution * legal_action
            action_distribution_sum = tf.expand_dims(tf.reduce_sum(action_distribution, axis=1), axis=-1) * np.ones((1, n_action), dtype=np.float32)
            action_distribution = action_distribution / action_distribution_sum
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=full_path)
        return params, action_distribution

    @staticmethod
    def build_critic_net(s_overview: Any, s_piece_pos: Any, s_atk_map: Any, outer_scope: str, name: str, trainable: bool = True) -> Any:
        if outer_scope and outer_scope.strip():
            full_path = outer_scope + '/' + name
        else:
            full_path = name
        with tf.variable_scope(name):
            input_tensor = Model.feature_extraction(s_overview, s_piece_pos, s_atk_map, trainable)
            l1 = tf.layers.Dense(
                units=512,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.constant_initializer(0.1),
                trainable=trainable,
            )(input_tensor)
            v = tf.layers.Dense(
                units=1,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.constant_initializer(0.1),
                trainable=trainable,
            )(l1)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=full_path)
        return params, v

    def act(self, state: Any, play: bool = False) -> int:
        s_overview, s_legal_action, s_piece_pos, s_atk_map = state  # not zip
        action = self.sess.run(self.a_dist, feed_dict={
            self.s_overview: np.expand_dims(s_overview, axis=0),
            self.s_legal_actions: np.expand_dims(s_legal_action, axis=0),
            self.s_piece_positions: np.expand_dims(s_piece_pos, axis=0),
            self.s_attack_map: np.expand_dims(s_atk_map, axis=0),
        })
        action = np.squeeze(action, axis=0)
        # print('X = ', action[action != 0], np.array(ChessEnv.init_actions())[action != 0])
        if play is False:
            return int(np.random.choice(action.shape[0], 1, p=action)[0])
        else:
            return int(np.argmax(action))

    def push_global(self, feed_dict) -> None:
        self.sess.run([self.push_a_params, self.push_c_params], feed_dict)

    def pull_global(self) -> None:
        self.sess.run([self.pull_a_params, self.pull_c_params])

    def save(self, path: str = './df_model/model.ckpt') -> str:
        save_path = self.saver.save(self.sess, path)
        return save_path

    def load(self, path: str = './df_model/model.ckpt') -> None:
        self.saver.restore(self.sess, path)
