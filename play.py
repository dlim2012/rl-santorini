from board_games.Santorini.board import GameBoard
from utils.utils import get_agent
from argparse import ArgumentParser


def parse():
    parser = ArgumentParser()

    parser.add_argument('--player1', '-p1', type=str, default='human',
                        help="Choose agent1: \'human\', \'random\', \'minimax1\', \'minimax2\', \'minimax3\', \'minimax4\', \'rl\' \
                        (trained model)")
    parser.add_argument('--player2', '-p2', type=str, default='rl',
                        help="Choose agent2")
    parser.add_argument('--player3', '-p3', type=str, default='none',
                        help="Choose agent3")
    parser.add_argument('--player4', '-p4', type=str, default='none',
                        help="Choose agent4")
    parser.add_argument('--player1_algorithm', '-a1', default='TRPO',
                        help="Algorithm used to train the model for player1: \'TRPO\', \'PPO\', \'A2C\'")
    parser.add_argument('--player2_algorithm', '-a2', default='TRPO',
                        help="Algorithm used to train the model for player2")
    parser.add_argument('--player3_algorithm', '-a3', default='TRPO',
                        help="Algorithm used to train the model for player3")
    parser.add_argument('--player4_algorithm', '-a4', default='TRPO',
                        help="Algorithm used to train the model for player4")
    parser.add_argument('--player1_ckpt_path', '-pt1', default='zoo/ckpt/TRPO_mixed/lr1.00e-02_run0',
                        help="Checkpoint path when agent1 is \'rl\'")
    parser.add_argument('--player2_ckpt_path', '-pt2', default='zoo/ckpt/TRPO_mixed/lr1.00e-02_run0',
                        help="Checkpoint path when agent2 is \'rl\'")
    parser.add_argument('--player3_ckpt_path', '-pt3', default='zoo/ckpt/TRPO_mixed/lr1.00e-02_run0',
                        help="Checkpoint path when agent3 is \'rl\'")
    parser.add_argument('--player4_ckpt_path', '-pt4', default='zoo/ckpt/TRPO_mixed/lr1.00e-02_run0',
                        help="Checkpoint path when agent4 is \'rl\'")
    parser.add_argument('--board_mode', '-bm', default='init-rand',
                        help="board mode: \'\' or \'init-rand\' (random worker initialization)")

    args = parser.parse_args()
    return args


def main():
    args = parse()

    player1 = get_agent(args.player1, args.player1_algorithm, args.player1_ckpt_path)
    player2 = get_agent(args.player2, args.player2_algorithm, args.player2_ckpt_path)
    player3 = get_agent(args.player3, args.player3_algorithm, args.player3_ckpt_path)
    player4 = get_agent(args.player4, args.player4_algorithm, args.player4_ckpt_path)

    agents = [player for player in [player1, player2, player3, player4] if player != 'none']

    game_board = GameBoard(
        agents=agents,
        print_simulation=True,
        mode=args.board_mode
    )
    game_board.print_occupied_spaces = False

    game_board.reset()
    play = game_board.play()
    for obs, reward, done, info in play:
        continue
    print("Total steps", info['n_turn'][0])
    for n_turn, event_list in info.items():
        if type(n_turn) == int:
            for event in event_list:
                print("turn %d: %s" % (n_turn, event))
    print("Winner team:", info['winner_team'])

if __name__ == '__main__':
    main()