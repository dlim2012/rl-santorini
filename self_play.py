from stable_baselines3 import A2C, DQN, PPO
from sb3_contrib import TRPO, ARS
import os

from board_games.Santorini.agents import RandomAgent, RLAgent
from board_games.Santorini.board import GameBoard
from board_games.CustomEnv import CustomEnv
from utils.evaluate import evaluate_policy

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from argparse import ArgumentParser

def parse():
    parser = ArgumentParser()

    parser.add_argument('--algorithm', type=str, default='PPO')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--invalid_action_reward', type=int, default=-10)

    parser.add_argument('--n_steps', type=int, default=int(1e5))
    parser.add_argument('--n_iteration', type=int, default=int(1e4))
    parser.add_argument('--n_eval_episodes', type=int, default=100)

    parser.add_argument('--update_interval', type=int, default=1)

    parser.add_argument('--print_simulation', action='store_true')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--save_name', type=str, default='default_name',
                        help='Models will be saved and loaded using this name.')
    parser.add_argument('--verbose', type=int, default=1)

    args = parser.parse_args()


    algorithms = {'PPO': PPO, 'A2C': A2C, 'TRPO': TRPO}#, 'ARS': ARS, 'DQN': DQN}
    args.algorithm = algorithms[args.algorithm]

    return args


def main(args):
    print('main')
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.save_name)

    agent = RLAgent(learning=True)
    opponent = RLAgent(learning=False)
    random_agent = RandomAgent()

    game_board = GameBoard(
        agents=[agent, opponent],
        learn_id=0,
        invalid_action_reward=args.invalid_action_reward
    )
    env = CustomEnv(game_board)
    env = Monitor(env)

    model = args.algorithm(
        'MlpPolicy',
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        verbose=2)

    env.enable_self_play(
        algorithm=args.algorithm,
        save_path=save_path,
        agent=agent,
        opponent=opponent,
        model=model,
        update_interval=args.update_interval
    )

    check_env(env)

    for i in range(int(1e4)):
        #print(env.game_board.agents, env.game_board.agents[0].model == env.game_board.agents[1].model)
        model.learn(total_timesteps=args.n_steps)

        print([(i + 1) * args.n_steps], '(vs. random)', end=' ')
        evaluate_policy(
            model,
            env,
            n_eval_episodes=args.n_eval_episodes,
            print_result=args.print_simulation,
            agents=[agent, random_agent],
            count_invalid_actions=int(1e3)
        )

        """
        # Versus self
        print([(i + 1) * args.n_steps], '( vs. self )', end=' ')
        evaluate_policy(
            model,
            env,
            n_eval_episodes=args.n_eval_episodes,
            print_result=args.print_simulation,
            agents=None,
            count_invalid_actions=int(1e3)
        )
        """


if __name__ == '__main__':
    args = parse()

    args.algorithm, args.lr, args.save_name = TRPO, 0.001, 'TRPO_dummy'#'TRPO_lr1e-3'
    #args.algorithm, args.lr, args.save_name = A2C, 0.03, 'A2C_dummy'#'TRPO_lr1e-3'
    #args.algorithm, args.lr, args.save_name = DQN, 0.01, 'DQN_lr1e-2'
    args.print_simulation=True
    #args.n_steps = 1000

    if args.save_name == 'default_name':
        args.save_name = '%s_lr%.2e' % (args.algorithm, args.lr)
    print(args.__dict__)


    main(args)
    #masked()