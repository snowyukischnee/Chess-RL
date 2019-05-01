import chess
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci('./stockfish/stockfish_win10_64')

board = chess.Board()
info = engine.analyse(board, chess.engine.Limit(time=0.1))
print(info)

board = chess.Board('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1')
info = engine.analyse(board, chess.engine.Limit(time=0.1))
print(info)

board = chess.Board('rnbqkbnr/1ppppppp/8/p7/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2')
info = engine.analyse(board, chess.engine.Limit(time=0.1))
print(info)

engine.quit()
