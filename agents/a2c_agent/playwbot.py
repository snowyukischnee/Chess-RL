import numpy as np

import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../../')))
from CP_CHESS.agents.a2c_agent.base_agent import BaseAgent
from CP_CHESS.agents.a2c_agent.base_config import BaseConfig
from CP_CHESS.env.environment import ChessEnv
from CP_CHESS.agents.a2c_agent.board2state import Board2State0 as board2state


class PlayConfig(object):
    def __init__(self) -> None:
        self.model_dir = './model'
        self.model_ver = 0


class PlayWBot(object):
    def __init__(self, config: PlayConfig) -> None:
        self.config = config
        self.bot = None
        self.env = None

    def init(self, config: BaseConfig, agent: BaseAgent) -> None:
        self.env = ChessEnv()
        config.n_action = len(self.env.actions)
        self.bot = agent(config)

    def load_model(self, model_dir: str, model_ver: int) -> None:
        self.config.model_dir = model_dir
        self.config.model_ver = model_ver
        self.bot.load_model(self.config.model_dir, self.config.model_ver)

    def play(self, player_is_white: bool = True) -> None:
        if player_is_white is True:
            state_type, state = self.env.reset(fen=None, board2state=board2state)
            print(self.env.board)  # print the board
            while True:
                action_str = input('Your move:')  # read the console input of the move
                if action_str == 'exit':
                    break
                elif action_str == 'pass':
                    self.env.pass_move()
                else:
                    action = self.env.actions.index(action_str)
                    state_type, next_state, reward, done, info = self.env.step(action, board2state=board2state)
                    state = next_state
                print(self.env.board)  # print the board
                if done:
                    print('Game over. Score: {}'.format(self.env.result))
                    break
                action = self.bot.action(state_type, state, play=True)
                print('Bot move {}'.format(self.env.actions[action]))
                state_type, next_state, reward, done, info = self.env.step(action, board2state=board2state)
                state = next_state
                print(self.env.board)  # print the board
                if done:
                    print('Game over. Score: {}'.format(self.env.result))
                    break
        else:
            state_type, state = self.env.reset(fen=None, board2state=board2state)
            while True:
                action = self.bot.action(state_type, state, play=True)
                print('Bot move {}'.format(self.env.actions[action]))
                state_type, next_state, reward, done, info = self.env.step(action, board2state=board2state)
                state = next_state
                print(self.env.board)  # print the board
                if done:
                    print('Game over. Score: {}'.format(self.env.result))
                    break
                action_str = input('Your move:')  # read the console input of the move
                if action_str == 'exit':
                    break
                elif action_str == 'pass':
                    self.env.pass_move()
                else:
                    action = self.env.actions.index(action_str)
                    state_type, next_state, reward, done, info = self.env.step(action, board2state=board2state)
                    state = next_state
                print(self.env.board)  # print the board
                if done:
                    print('Game over. Score: {}'.format(self.env.result))
                    break