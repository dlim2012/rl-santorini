from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_checker import check_env
from argparse import ArgumentParser
from sb3_contrib import TRPO, ARS
import os
import time, math
from matplotlib import pyplot as plt
import numpy as np

from board_games.Santorini.agents import RandomAgent, RLAgent, MiniMaxAgent
from board_games.Santorini.board import GameBoard
from board_games.CustomEnv import CustomEnv
from utils.evaluate import evaluate_policy
from utils.tools import time_hr_min_sec


def parse():
    parser = ArgumentParser()

    parser.add_argument('--algorithm', type=str, default='PPO')
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--invalid_action_reward', type=int, default=-10)

    parser.add_argument('--n_steps', type=int, default=int(1e5))
    parser.add_argument('--n_iter', type=int, default=int(3e2))
    parser.add_argument('--n_eval_episodes', type=int, default=100)

    parser.add_argument('--update_interval', type=int, default=1)

    parser.add_argument('--print_simulation', action='store_true')
    parser.add_argument('--plt_save_dir', type=str, default='plt')
    parser.add_argument('--ckpt_save_dir', type=str, default='checkpoints')
    parser.add_argument('--save_name', type=str, default='default_name',
                        help='Models will be saved and loaded using this name.')
    parser.add_argument('--verbose', type=int, default=1)

    args = parser.parse_args()

    assert args.save_name + '.zip' not in os.listdir(args.ckpt_save_dir)

    algorithms = {'PPO': PPO, 'A2C': A2C, 'TRPO': TRPO}
    args.algorithm = algorithms[args.algorithm]

    return args


def main(args):
    start = time.time()

    print('main')
    os.makedirs(args.ckpt_save_dir, exist_ok=True)
    os.makedirs(args.plt_save_dir, exist_ok=True)
    save_path = os.path.join(args.ckpt_save_dir, args.save_name)

    agent = RLAgent(learning=True)
    opponent = RLAgent(learning=False)

    opponents = [
        RandomAgent(),
        MiniMaxAgent(maxDepth=1),
        MiniMaxAgent(maxDepth=2),
        MiniMaxAgent(maxDepth=3)
    ]
    n_eval_episodes_list = [
        args.n_eval_episodes,
        args.n_eval_episodes,
        math.ceil(args.n_eval_episodes / 5),
        math.ceil(args.n_eval_episodes / 25)
    ]

    game_board = GameBoard(
        agents=[agent, opponent],
        learn_id=0,
        invalid_action_reward=args.invalid_action_reward
    )
    env = CustomEnv(game_board)

    model = args.algorithm(
        'MlpPolicy',
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        verbose=1)

    env.enable_self_play(
        algorithm=args.algorithm,
        save_path=None,
        agent=agent,
        opponent=opponent,
        model=model,
        update_interval=args.update_interval
    )

    check_env(env)

    results = [[] for _ in range(4)]
    for i in range(args.n_iter):
        model.learn(total_timesteps=args.n_steps)
        print('iteration %d (each iteration: %d steps)' % (i+1, args.n_steps))

        for i in range(4):
            average_rewards, counts = evaluate_policy(
                model,
                env,
                n_eval_episodes=n_eval_episodes_list[i],
                print_result=args.print_simulation,
                agents=[agent, opponents[i]],
                count_invalid_actions=int(1e3)
            )

            results[i].append(average_rewards)


        print('%d hr %d min %d sec' % time_hr_min_sec(time.time() - start))
        print()

        model.save(save_path)

    plt.figure(figsize=(8, 6))
    for i in range(4):
        plt.plot(np.arange(args.n_iter), results[i])
    plt.savefig(os.path.join(args.plt_save_dir, args.save_name))
    plt.clf()
    plt.close()

if __name__ == '__main__':
    args = parse()

    main(args)