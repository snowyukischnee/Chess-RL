import numpy as np

import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../')))

from CP_CHESS.agents.base_agent.agent import BaseAgent
from CP_CHESS.agents.base_agent.config import BaseConfig

from CP_CHESS.agents.my_agent.memory import Memory
from CP_CHESS.env.environment import ChessEnv
from CP_CHESS.utils.board2state import Board2State0 as board2state
from CP_CHESS.agents.my_agent.config import Config
from CP_CHESS.agents.my_agent.agent import Agent


class SelfPlayConfig(object):
    def __init__(self):
        self.n_episodes = 50
        self.max_steps = 277  # the largest number of moves in a game of chess in record
        self.model_dir = './model'
        self.model_ver = 0


class SelfPlay(object):
    def __init__(self, config: SelfPlayConfig):
        self.config = config
        self.current_player = None
        self.target_player = None
        self.env = None
        self.memory = Memory(self.config.max_steps)

    def init(self, config: BaseConfig, agent: BaseAgent):
        self.env = ChessEnv()
        config.n_action = len(self.env.actions)
        self.current_player = agent(config)
        self.target_player = agent(config)
        self.memory.clear()

    def process(self, opponent_is_white: bool = False):
        actions = np.identity(len(self.env.actions))
        self.target_player.load_model(self.config.model_dir, self.config.model_ver)
        if opponent_is_white is True:
            for episode in range(self.config.n_episodes):
                self.memory.clear()
                timestep = 0
                state_type, state = self.env.reset(fen=None, board2state=board2state)
                while True:
                    timestep += 1
                    # opponent's turn
                    action = self.target_player.action(state_type, state)
                    state_type, next_state, reward, done, info = self.env.step(action, board2state=board2state)
                    state = next_state
                    if done or timestep > self.config.max_steps:
                        self.current_player.model.learn(self.memory.sample(self.config.max_steps, continuous=True))
                        break
                    # player's turn
                    action = self.current_player.action(state_type, state)
                    state_type, next_state, reward, done, info = self.env.step(action, board2state=board2state)
                    self.memory.add((state, actions[action], next_state, reward))
                    state = next_state
                    if done or timestep > self.config.max_steps:
                        self.current_player.model.learn(self.memory.sample(self.config.max_steps, continuous=True))
                        break
        else:
            for episode in range(self.config.n_episodes):
                self.memory.clear()
                timestep = 0
                state_type, state = self.env.reset(fen=None, board2state=board2state)
                while True:
                    timestep += 1
                    # player's turn
                    action = self.current_player.action(state_type, state)
                    state_type, next_state, reward, done, info = self.env.step(action, board2state=board2state)
                    self.memory.add((state, actions[action], next_state, reward))
                    state = next_state
                    if done or timestep > self.config.max_steps:
                        self.current_player.model.learn(self.memory.sample(self.config.max_steps, continuous=True))
                        break
                    # opponent's turn
                    action = self.target_player.action(state_type, state)
                    state_type, next_state, reward, done, info = self.env.step(action, board2state=board2state)
                    state = next_state
                    if done or timestep > self.config.max_steps:
                        self.current_player.model.learn(self.memory.sample(self.config.max_steps, continuous=True))
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
