from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_checker import check_env
from argparse import ArgumentParser
from sb3_contrib import TRPO, ARS
import os
import time

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
    parser.add_argument('--n_iter', type=int, default=int(1e2))
    parser.add_argument('--n_eval_episodes', type=int, default=100)

    parser.add_argument('--update_interval', type=int, default=1)

    parser.add_argument('--print_simulation', action='store_true')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--save_name', type=str, default='default_name',
                        help='Models will be saved and loaded using this name.')
    parser.add_argument('--verbose', type=int, default=1)

    args = parser.parse_args()

    assert args.save_name + '.zip' not in os.listdir(args.save_dir)

    algorithms = {'PPO': PPO, 'A2C': A2C, 'TRPO': TRPO}#, 'ARS': ARS, 'DQN': DQN}
    args.algorithm = algorithms[args.algorithm]

    return args


def main(args):
    start = time.time()

    print('main')
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.save_name)

    agent = RLAgent(learning=True)
    opponent = RLAgent(learning=False)
    random_agent = RandomAgent()
    minimax1_agent = MiniMaxAgent(maxDepth=1)
    minimax2_agent = MiniMaxAgent(maxDepth=2)
    minimax3_agent = MiniMaxAgent(maxDepth=3)

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

    for i in range(args.n_iter):
        #print(env.game_board.agents, env.game_board.agents[0].model == env.game_board.agents[1].model)
        model.learn(total_timesteps=args.n_steps)
        print('iteration %d (each iteration: %d steps)' % (i+1, args.n_steps))

        evaluate_policy(
            model,
            env,
            n_eval_episodes=args.n_eval_episodes,
            print_result=args.print_simulation,
            agents=[agent, opponent],
            count_invalid_actions=int(1e3)
        )
        evaluate_policy(
            model,
            env,
            n_eval_episodes=100,
            print_result=args.print_simulation,
            agents=[agent, minimax1_agent],
            count_invalid_actions=int(1e3)
        )
        evaluate_policy(
            model,
            env,
            n_eval_episodes=20,
            print_result=args.print_simulation,
            agents=[agent, minimax2_agent],
            count_invalid_actions=int(1e3)
        )
        evaluate_policy(
            model,
            env,
            n_eval_episodes=4,
            print_result=args.print_simulation,
            agents=[agent, minimax3_agent],
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

        print('%d hr %d min %d sec' % time_hr_min_sec(time.time() - start))
        print()

if __name__ == '__main__':
    args = parse()

    args.algorithm, args.lr, args.save_name = TRPO, 0.001, 'TRPO_dummy'#'TRPO_lr1e-3'
    #args.algorithm, args.lr, args.save_name = A2C, 0.03, 'A2C_dummy2'#'TRPO_lr1e-3'
    #args.algorithm, args.lr, args.save_name = PPO, 0.03, 'PPO_dummy-'#'TRPO_lr1e-3'
    #args.algorithm, args.lr, args.save_name = DQN, 0.01, 'DQN_lr1e-2'
    args.print_simulation=True
    args.n_steps = 100000

    if args.save_name == 'default_name':
        args.save_name = '%s_lr%.2e' % (args.algorithm, args.lr)
    print(args.__dict__)


    #assert args.save_name + '.zip' not in os.listdir(args.save_dir)

    main(args)
    #masked()