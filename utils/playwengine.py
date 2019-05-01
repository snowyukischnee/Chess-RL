import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../')))
from CP_CHESS.env.stockfish_env import ChessEnvWrapper


class PlayWEngine(object):
    def __init__(self, bin_path: str, timelimit: float):
        self.env = ChessEnvWrapper(bin_path, timelimit)

    def play(self, player_white_pieces: bool = True) -> None:
        done = False
        print(self.env.env.board)
        _, _ = self.env.reset(player_white_pieces=player_white_pieces)
        print(self.env.env.board)
        while True:
            action_str = input('Your move:')
            if action_str == 'exit':
                self.env.engine.quit()
                break
            elif action_str == 'pass':
                self.env.env.pass_move()
            else:
                action = self.env.env.actions.index(action_str)
                _, _, reward, done, _ = self.env.step(action)
            print(self.env.env.board)
            if done:
                print('Reward: {}'.format(reward))
                self.env.engine.quit()
                break