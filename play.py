from board_games.Santorini.board import GameBoard
from board_games.Santorini.agents import get_agent
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
    parser.add_argument('--player1_ckpt_path', '-pt1', default='',
                        help="Checkpoint path when player1 is \'rl\'")
    parser.add_argument('--player2_ckpt_path', '-pt2', default='',
                        help="Checkpoint path when player2 is \'rl\'")
    parser.add_argument('--player3_ckpt_path', '-pt3', default='',
                        help="Checkpoint path when player3 is \'rl\'")
    parser.add_argument('--player4_ckpt_path', '-pt4', default='',
                        help="Checkpoint path when player4 is \'rl\'")
    parser.add_argument('--board_mode', '-bm', default='init-rand',
                        help="board mode: \'\' or \'init-rand\' (random worker initialization)")

    args = parser.parse_args()

    args.players = [
        (args.player1, args.player1_algorithm, args.player1_ckpt_path),
        (args.player2, args.player2_algorithm, args.player2_ckpt_path),
        (args.player3, args.player3_algorithm, args.player3_ckpt_path),
        (args.player4, args.player4_algorithm, args.player4_ckpt_path),
    ]


    return args


def main():
    args = parse()

    agents = [player for player in
        [get_agent(args.players[i][0], args.players[i][1], args.players[i][2]) for i in range(4)]
              if player != 'none'
    ]

    for agent in agents:
        if agent.agent_type == "rl" and agent.model == None:
            if len(agents) == 2 and args.board_mode == 'init-rand':
                agent.set_model('TRPO', 'zoo/ckpt/TRPO_mixed_2p/lr1.00e-02_run0')
            else:
                raise ValueError("No model provided for RL agent")

    GameBoard(
        agents=agents,
        print_simulation=2,
        mode=args.board_mode,
        use_gui=True,
        delay=0.0
    )


if __name__ == '__main__':
    main()