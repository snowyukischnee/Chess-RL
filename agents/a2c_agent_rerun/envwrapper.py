from typing import Any

import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../../')))
from CP_CHESS.agents.a2c_agent_rerun.model import Model
from CP_CHESS.agents.a2c_agent_rerun.board2state import Board2State0 as board2state
from CP_CHESS.env.environment import ChessEnv


class ChessEnvWrapper(object):
    def __init__(self, opponent: Model):
        self.env = ChessEnv()
        self.opponent = opponent
        self.deterministic_policy = False

    def reset(self, player_white_pieces: bool = True) -> Any:
        if player_white_pieces is True:
            state_type, player_state = self.env.reset(fen=None, board2state=board2state)
            return state_type, player_state
        else:
            state_type, opponent_state = self.env.reset(fen=None, board2state=board2state)
            opponent_action = self.opponent.act(opponent_state, play=self.deterministic_policy)
            state_type, opponent_next_state, opponent_reward, done, info = self.env.step(opponent_action, board2state=board2state)
            player_state = opponent_next_state
            return state_type, player_state

    def step(self, action: int) -> Any:
        player_action = action
        state_type, player_next_state, player_reward, done, info = self.env.step(player_action, board2state=board2state)
        if done:
            reward = player_reward
            player_state = player_next_state
            return state_type, player_state, reward, done, info
        opponent_state = player_next_state
        opponent_action = self.opponent.act(opponent_state, play=self.deterministic_policy)
        state_type, opponent_next_state, opponent_reward, done, info = self.env.step(opponent_action, board2state=board2state)
        reward = -opponent_reward
        player_state = opponent_next_state
        return state_type, player_state, reward, done, info