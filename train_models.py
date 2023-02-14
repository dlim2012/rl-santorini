from copy import deepcopy

from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO
import os
from tensorboardX import SummaryWriter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import torch
from argparse import ArgumentParser

from board_games.CustomEnv import CustomEnv
from utils.evaluate import evaluate_policy
from utils.utils import get_agent


def parse():
    parser = ArgumentParser()

    parser.add_argument('--board', '-b', type=str, default='Santorini',
                        help='Choose a board game: Santorini')
    parser.add_argument('--board_mode', '-bm', type=str, default='init-rand',
                        help='Santorini: init-rand(random initialization for worker)')
    parser.add_argument('--algorithm', '-a', type=str, default='TRPO',
                        help='Choose an algorithm: PPO, TRPO, A2C')
    parser.add_argument('--mode', '-m', type=str, default='learn',
                        help="""
                        Choose a mode for train:\n
                        \tlearn (train versus a minimax agent with max depth 2),\n
                        \tself_play (train against itself)
                        \tmixed (self_play and minimax agent with max depth 2)
                        """)
    parser.add_argument('--n_runs', '-n', type=int, default=5,
                        help='Number of models to train: (default 5)')
    parser.add_argument('--lr', '-lr', type=float, default=1e-2,
                        help='Choose a learning rate for training.')
    parser.add_argument('--invalid_action_reward', '-i', type=int, default=-10,
                        help='Rewards for invalid actions')
    parser.add_argument('--n_steps', '-s', type=int, default=int(1e5),
                        help='hyperparameter n_steps for algorithms (PPO, A2C, TRPO)')
    parser.add_argument('--n_iter', '-it', type=int, default=int(1e4),
                        help='Number of iterations of training (total steps: n_iter * n_steps)')
    parser.add_argument('--update_interval', '-u', type=int, default=1,
                        help='update interval for self-play')
    parser.add_argument('--eval_mode', '-e', type=str, default='default',
                        help='Argument for choosing opponents for evaluation')

    parser.add_argument('--player3', '-p3', type=str, default='none',
                        help="Choose agent3")
    parser.add_argument('--player4', '-p4', type=str, default='none',
                        help="Choose agent4")
    parser.add_argument('--player3_algorithm', '-a3', default='TRPO',
                        help="Algorithm used to train the model for player3: \'TRPO\', \'PPO\', \'A2C\'")
    parser.add_argument('--player4_algorithm', '-a4', default='TRPO',
                        help="Algorithm used to train the model for player4")
    parser.add_argument('--player3_ckpt_path', '-pt3', default='zoo/ckpt/TRPO_mixed/lr1.00e-02_run0',
                        help="Checkpoint path when agent3 is \'rl\'")
    parser.add_argument('--player4_ckpt_path', '-pt4', default='zoo/ckpt/TRPO_mixed/lr1.00e-02_run0',
                        help="Checkpoint path when agent4 is \'rl\'")

    parser.add_argument('--test_run', '-t', action='store_true',
                        help='Train one model without multi-processing for debugging purpose')
    parser.add_argument('--name', '-name', type=str, default="",
                        help="This will be appended to the directory names for logs and checkpoints")

    args = parser.parse_args()
    return args


def learn(args, ckpt_path, log_path):

    torch.set_num_threads(1)

    if args.algorithm == 'PPO':
        algorithm = PPO
    elif args.algorithm == 'A2C':
        algorithm = A2C
    elif args.algorithm == 'TRPO':
        algorithm = TRPO
    else:
        raise ValueError()

    if args.board == 'Santorini':
        from board_games.Santorini import board, agents, tools
    else:
        raise ValueError()

    agent = agents.RLAgent(learning=True)
    weights = None
    if args.mode == 'learn':
        opponents = [agents.MiniMaxAgent(max_depth=2)]
    elif args.mode == 'self_play':
        opponents = [agents.RLAgent(learning=False)]
    elif args.mode == 'mixed':
        opponents = [agents.RLAgent(learning=False), agents.MiniMaxAgent(max_depth=2)]
        weights = [2, 8]
    else:
        raise ValueError()

    eval_opponents = tools.get_eval_opponents(args.eval_mode)

    agents = [agent, opponents[0]]
    player3 = get_agent(args.player3, args.player3_algorithm, args.player3_ckpt_path)
    player4 = get_agent(args.player4, args.player4_algorithm, args.player4_ckpt_path)

    for player in [player3, player4]:
        if player != 'none':
            agents.append(player)
    print("Number of agents:", len(agents))

    game_board = board.GameBoard(
        agents=agents,
        learn_id=0,
        invalid_action_reward=args.invalid_action_reward,
        mode=args.board_mode
    )
    env = CustomEnv(game_board)

    model = algorithm(
        'MlpPolicy',
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        verbose=0,
        device='cpu'
    )

    if args.mode in ['self_play', 'mixed']:
        env.enable_self_play(
            algorithm=algorithm,
            model=model,
            save_path=None,
            agents=agents,
            opponents=opponents,
            weights=weights,
            update_interval=args.update_interval
        )

    writer = SummaryWriter(log_path)
    for i in range(args.n_iter):
        print([i], end='')

        model.learn(total_timesteps=args.n_steps)
        for name, opponent, n_eval_episodes in eval_opponents:
            average_rewards, counts = evaluate_policy(
                model,
                env,
                n_eval_episodes=n_eval_episodes,
                print_result=args.test_run,
                agents=[agent] + [deepcopy(opponent) for _ in range(len(agents) -1)],
                count_invalid_actions=int(1e3)
            )

            writer.add_scalar(name, average_rewards, args.n_steps * (i + 1))
            writer.add_scalar('invalid_action_count_%s' % name, counts['invalid_action_count'],
                              args.n_steps * (i + 1))

        writer.flush()
        model.save(ckpt_path)

    writer.close()


def main(args):
    save_dir = 'zoo'
    save_name = args.algorithm + '_' + args.mode
    if args.name:
        save_name += '_' + args.name
    ckpt_dir = os.path.join(save_dir, 'ckpt', save_name)
    log_dir = os.path.join(save_dir, 'logs', save_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if args.test_run:
        name = 'test'
        learn(args, os.path.join(ckpt_dir, name), os.path.join(log_dir, name))
        return

    pool = ProcessPoolExecutor(max_workers=cpu_count())
    for i in range(args.n_runs):
        name = 'lr%.2e_run%d' % (args.lr, i)
        pool.submit(
            learn,
            args,
            os.path.join(ckpt_dir, name),
            os.path.join(log_dir, name)
        )


if __name__ == '__main__':
    args = parse()
    main(args)

