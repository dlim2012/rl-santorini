from board_games.Santorini.agents import HumanAgent, RLAgent, MiniMaxAgent
from board_games.Santorini.board import GameBoard

from stable_baselines3 import PPO, A2C
from sb3_contrib import TRPO
from argparse import ArgumentParser


def parse():
    parser = ArgumentParser()

    parser.add_argument('--algorithm', '-a', type=str, default='TRPO')
    parser.add_argument('--opponent', '-o', type=str, default='human')
    parser.add_argument('--depth', '-d', type=int, default=3)
    parser.add_argument('--ckpt_path', '-ckpt', default='zoo/ckpt/TRPO_mixed_2p/1')
    parser.add_argument('--board_mode', '-bm', default='init-rand',
                        help="board mode: \'\' or \'init-rand\' (random worker initialization)")

    args = parser.parse_args()
    algorithms = {'TRPO': TRPO, 'A2C': A2C, 'PPO': PPO}
    args.algorithm = algorithms[args.algorithm]
    return args


def main():
    args = parse()

    ####
    args.depth = 3
    ####

    agent = RLAgent()
    agent.model = args.algorithm.load(args.ckpt_path)

    n_games = 100
    for depth in [1, 2, 3]:
        win = 0
        print('depth', depth)
        opponent = MiniMaxAgent(max_depth=depth)

        game_board = GameBoard(
            agents=[agent, opponent],
            mode=args.board_mode,
            # print_simulation=True
        )
        for i in range(n_games):
            game_board.reset()
            play = game_board.play()
            ret = next(play)[-1]['winner_team']
            print(ret, end='')
            if ret == 0:
                win += 1
        print('\n', win)
        print("win_ratio: %.2f" % (win / n_games * 100) + "%");

if __name__ == '__main__':
    main()