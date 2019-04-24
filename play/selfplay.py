import numpy as np

import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../')))
from CP_CHESS.agents.base_agent.agent import BaseAgent
from CP_CHESS.agents.base_agent.config import BaseConfig
from CP_CHESS.agents.a2c_agent.memory import Memory
from CP_CHESS.env.environment import ChessEnv
from CP_CHESS.agents.a2c_agent.board2state import Board2State0 as board2state
from CP_CHESS.agents.a2c_agent.config import Config
from CP_CHESS.agents.a2c_agent.agent import Agent


class SelfPlayConfig(object):
    def __init__(self) -> None:
        self.n_episodes = 10
        self.update_interval = 32
        self.max_steps = 32 * 20
        self.model_dir = './model'
        self.model_ver = 0


class SelfPlay(object):
    def __init__(self, config: SelfPlayConfig) -> None:
        self.config = config
        self.current_player = None
        self.target_player = None
        self.env = None
        self.memory = Memory(self.config.update_interval)

    def init(self, config: BaseConfig, agent: BaseAgent, gpu_idx: int = None) -> None:
        self.env = ChessEnv()
        config.n_action = len(self.env.actions)
        self.current_player = agent(config, gpu_idx)
        self.target_player = agent(config, gpu_idx)
        self.memory.clear()

    def resume(self, model_dir: str, model_ver: int) -> None:
        self.config.model_dir = model_dir
        self.config.model_ver = model_ver
        self.current_player.load_model(self.config.model_dir, self.config.model_ver)

    def process(self, opponent_is_white: bool = False) -> None:
        actions = np.identity(len(self.env.actions))
        self.target_player.load_model(self.config.model_dir, self.config.model_ver)
        if opponent_is_white is True:
            for episode in range(self.config.n_episodes):
                self.memory.clear()
                timestep = 0
                interval_clock = 0
                # opponent play first
                state_type, opponent_state = self.env.reset(fen=None, board2state=board2state)
                opponent_action = self.target_player.action(state_type, opponent_state)
                state_type, opponent_next_state, opponent_reward, done, info = self.env.step(opponent_action, board2state=board2state)
                player_state = opponent_next_state
                print('[opponent]state: {}, action: {}, done: {}'.format(self.env.board.fen(), self.env.actions[opponent_action], done))
                while True:
                    timestep += 1
                    interval_clock += 1
                    # player's turn
                    player_action = self.current_player.action(state_type, player_state)
                    state_type, player_next_state, player_reward, done, info = self.env.step(player_action, board2state=board2state)
                    opponent_state = player_next_state
                    print('[player]state: {}, action: {}, done: {}'.format(self.env.board.fen(), self.env.actions[player_action], done))
                    if done:
                        reward = player_reward
                        print('[RECAP]state: {}, action: {}, reward: {}, done: {}'.format(self.env.board.fen(), self.env.actions[player_action], reward, done))
                        self.memory.add((player_state, actions[player_action], player_next_state, player_reward))
                        interval_clock = 0
                        self.current_player.model.learn(self.memory.sample(self.config.max_steps, continuous=True))
                    if done:
                        break
                    # opponent's turn
                    opponent_action = self.target_player.action(state_type, opponent_state)
                    state_type, opponent_next_state, opponent_reward, done, info = self.env.step(opponent_action, board2state=board2state)
                    reward = -opponent_reward
                    self.memory.add((player_state, actions[player_action], opponent_next_state, reward))
                    player_state = opponent_next_state
                    print('[opponent]state: {}, action: {}, done: {}'.format(self.env.board.fen(), self.env.actions[opponent_action], done))
                    print('[RECAP]state: {}, action: {}, reward: {}, done: {}'.format(self.env.board.fen(), self.env.actions[player_action], reward, done))
                    if done or interval_clock > self.config.update_interval or timestep > self.config.max_steps:
                        interval_clock = 0
                        self.current_player.model.learn(self.memory.sample(self.config.max_steps, continuous=True))
                    if done or timestep > self.config.max_steps:
                        break
        else:
            for episode in range(self.config.n_episodes):
                self.memory.clear()
                timestep = 0
                interval_clock = 0
                state_type, player_state = self.env.reset(fen=None, board2state=board2state)
                while True:
                    timestep += 1
                    interval_clock += 1
                    # player's turn
                    player_action = self.current_player.action(state_type, player_state)
                    state_type, player_next_state, player_reward, done, info = self.env.step(player_action, board2state=board2state)
                    opponent_state = player_next_state
                    print('[player]state: {}, action: {}, done: {}'.format(self.env.board.fen(), self.env.actions[player_action], done))
                    if done:
                        reward = player_reward
                        print('[RECAP]state: {}, action: {}, reward: {}, done: {}'.format(self.env.board.fen(), self.env.actions[player_action], reward, done))
                        self.memory.add((player_state, actions[player_action], player_next_state, player_reward))
                        interval_clock = 0
                        self.current_player.model.learn(self.memory.sample(self.config.max_steps, continuous=True))
                    if done:
                        break
                    # opponent's turn
                    opponent_action = self.target_player.action(state_type, opponent_state)
                    state_type, opponent_next_state, opponent_reward, done, info = self.env.step(opponent_action, board2state=board2state)
                    reward = -opponent_reward
                    self.memory.add((player_state, actions[player_action], opponent_next_state, reward))
                    print('[opponent]state: {}, action: {}, done: {}'.format(self.env.board.fen(), self.env.actions[opponent_action], done))
                    print('[RECAP]state: {}, action: {}, reward: {}, done: {}'.format(self.env.board.fen(), self.env.actions[player_action], reward, done))
                    player_state = opponent_next_state
                    if done or interval_clock > self.config.update_interval or timestep > self.config.max_steps:
                        interval_clock = 0
                        self.current_player.model.learn(self.memory.sample(self.config.max_steps, continuous=True))
                    if done or timestep > self.config.max_steps:
                        break
        save_path = self.current_player.save_model()
        print('model saved at {}'.format(save_path))
        self.config.model_ver += 1


if __name__ == '__main__':
    env = ChessEnv()
    config = Config()
    config.n_action = len(env.actions)
    _, a = env.reset(fen=None, board2state=board2state)
    tp, a, b, c, d = env.step(928, board2state=board2state)
    ag = Agent(config)
    x = ag.action(tp, a)
