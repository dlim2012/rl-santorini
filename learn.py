from stable_baselines3 import A2C, DQN, PPO
from sb3_contrib import TRPO
import os
from argparse import ArgumentParser

from board_games.Santorini.agents import RandomAgent, RLAgent, MiniMaxAgent
from board_games.Santorini.board import GameBoard
from board_games.CustomEnv import CustomEnv
from utils.evaluate import evaluate_policy

from stable_baselines3.common.env_checker import check_env

def parse():
    parser = ArgumentParser()

    parser.add_argument('--algorithm', type=str, default='PPO')
    parser.add_argument('--invalid_action_reward', type=int, default=-10)
    parser.add_argument('--depth', type=int, default=1)

    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--n_steps', type=int, default=int(1e5))
    parser.add_argument('--n_iteration', type=int, default=int(1e4))
    parser.add_argument('--n_eval_episodes', type=int, default=100)
    parser.add_argument('--print_simulation', action='store_true')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--save_name', type=str, default='random')

    args = parser.parse_args()
    algorithms = {'PPO': PPO, 'A2C': A2C, 'TRPO': TRPO, 'DQN': DQN}
    args.algorithm = algorithms[args.algorithm]

    return args


def main(args):
    print('main')
    save_path = os.path.join(args.save_dir, args.save_name)
    os.makedirs(save_path, exist_ok=True)

    agent = RLAgent(learning=True)
    if args.depth == 0:
        opponent = RandomAgent()
    else:
        opponent = MiniMaxAgent(maxDepth=args.depth)

    n_steps = args.n_steps

    game_board = GameBoard(
        agents=[agent, opponent],
        learn_id=0,
        invalid_action_reward=args.invalid_action_reward
    )
    env = CustomEnv(game_board)

    check_env(env)
    if args.algorithm in [PPO, A2C, TRPO]:
        model = args.algorithm(
            'MlpPolicy',
            env,
            learning_rate=args.lr,
            #learning_starts=10000,
            n_steps=n_steps,
            verbose=1,
        )
    for i in range(int(1e4)):
        model.learn(total_timesteps=n_steps)

        print([(i+1) * n_steps], end=' ')
        evaluate_policy(model, env, n_eval_episodes=100, print_result=True, count_invalid_actions=1000)
        if i > 0:
            try:
                os.remove(save_name)
            except:
                pass
        save_name = os.path.join(save_path, '%.3e' % ((i+1) * n_steps))
        model.save(save_name)

if __name__ == '__main__':
    args = parse()
    #args = Namespace(**{'algorithm': 'PPO', 'lr': 0.03})
    #args = Namespace(**{'algorithm': 'A2C', 'lr': 0.03})
    args.algorithm, args.lr = TRPO, 0.001
    #args.save_name = 'TRPO_Random'
    args.n_steps, args.n_iter = 10000, 1000
    args.depth = 2
    print(args)

    main(args)
    #masked()