from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO, ARS
import os
from stable_baselines3.common.env_checker import check_env
from tensorboardX import SummaryWriter

from board_games.Santorini.agents import RandomAgent, RLAgent, MiniMaxAgent
from board_games.Santorini.board import GameBoard
from board_games.CustomEnv import CustomEnv
from utils.evaluate import evaluate_policy
from concurrent.futures import ProcessPoolExecutor

from argparse import ArgumentParser
import torch


def parse():
    parser = ArgumentParser()

    parser.add_argument('--algorithm', type=str, default='TRPO')
    parser.add_argument('--mode', type=str, default='learn')
    parser.add_argument('--n_runs', type=int, default=5)
    args = parser.parse_args()
    return args

def learn(algorithm, lr, total_timesteps, ckpt_path, log_path, mode):

    torch.set_num_threads(1)
    n_steps = int(1e5)


    writer = SummaryWriter(log_path)

    agent = RLAgent(learning=True)
    if mode == 'learn':
        opponent = MiniMaxAgent(maxDepth=2)
    elif mode == 'self_play':
        opponent = RLAgent(learning=False)
    else:
        raise ValueError()

    opponents = [
        ('random', RandomAgent(), 100),
        ('minimax1', MiniMaxAgent(maxDepth=1), 100),
        ('minimax2', MiniMaxAgent(maxDepth=2), 20),
        ('minimax3', MiniMaxAgent(maxDepth=3), 10),
    ]

    game_board = GameBoard(
        agents=[agent, opponent],
        learn_id=0,
        invalid_action_reward=-10
    )
    env = CustomEnv(game_board)

    model = algorithm(
        'MlpPolicy',
        env,
        learning_rate=lr,
        n_steps=n_steps,
        verbose=0,
        device='cpu'
    )

    if mode == 'self_play':
        env.enable_self_play(
            algorithm=algorithm,
            save_path=None,
            agent=agent,
            opponent=opponent,
            model=model,
            update_interval=1,
            save_interval=0
        )

    check_env(env)

    n_iter = total_timesteps // n_steps
    for i in range(n_iter):
        print([i], end='')

        model.learn(total_timesteps=n_steps)

        for name, opponent, n_eval_episodes in opponents:
            average_rewards, counts = evaluate_policy(
                model,
                env,
                n_eval_episodes=n_eval_episodes,
                print_result=False,
                agents=[agent, opponent],
                count_invalid_actions=int(1e3)
            )

            writer.add_scalar(name, average_rewards, n_steps * (i + 1))
            writer.add_scalar('invalid_action_count_%s' % name, counts['invalid_action_count'],
                              n_steps * (i + 1))

        model.save(ckpt_path)

    writer.close()

def main(args):
    print('main')


    if args.algorithm == 'PPO':
        learning_rates = [1e-1, 1e-2, 1e-3, 1e-4]
        total_timesteps = int(2e7)
        algorithm = PPO
    elif args.algorithm == 'A2C':
        learning_rates = [1e-1, 1e-2, 1e-3, 1e-4]
        total_timesteps = int(2e7)
        algorithm = A2C
    elif args.algorithm == 'TRPO':
        learning_rates = [1e-1, 1e-2, 1e-3, 1e-4]
        total_timesteps = int(2e7)
        algorithm = TRPO
    else:
        raise ValueError()

    pool = ProcessPoolExecutor(max_workers=40)

    for lr in learning_rates:
        for i in range(args.n_runs):
            pool.submit(
                learn,
                algorithm,
                lr,
                total_timesteps,
                os.path.join('model', 'ckpt', args.algorithm + '_' + args.mode, 'lr%.2e_run%d' % (lr, i)),
                os.path.join('model', 'logs', args.algorithm + '_' + args.mode, 'lr%.2e_run%d' % (lr, i)),
                args.mode
            )

if __name__ == '__main__':
    args = parse()
    print(args)
    main(args)

