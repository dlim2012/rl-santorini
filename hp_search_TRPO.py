from stable_baselines3 import A2C, DQN, PPO
from sb3_contrib import TRPO, ARS
import os
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from argparse import ArgumentParser
import time
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from board_games.Santorini.agents import RandomAgent, RLAgent
from board_games.Santorini.board import GameBoard
from board_games.CustomEnv import CustomEnv
from utils.evaluate import evaluate_policy
from utils.tools import time_hr_min_sec, Namespace
from concurrent.futures import ProcessPoolExecutor


def learn(lr, invalid_action_reward, n_steps, cg_damping, n_critic_updates, gae_lambda):
    start = time.time()

    model_save_dir = 'hp_search/checkpoints/PPO'
    fig_save_dir = 'hp_search/results/PPO'
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(fig_save_dir, exist_ok=True)

    save_name = 'lr%.2e_ir%.2e_ns%.2e_cd%.2e_cu%.2e_gl%.2e' % \
                (lr, invalid_action_reward, n_steps, cg_damping, n_critic_updates, gae_lambda)
    print(save_name)

    agent = RLAgent(learning=True)
    opponent = RLAgent(learning=False)
    random_agent = RandomAgent()

    game_board = GameBoard(
        agents=[agent, opponent],
        learn_id=0,
        invalid_action_reward=invalid_action_reward
    )
    env = CustomEnv(game_board)
    env = Monitor(env)

    model = TRPO(
        'MlpPolicy',
        env,
        learning_rate=lr,
        n_steps=n_steps,
        cg_damping=cg_damping,
        n_critic_updates=n_critic_updates,
        gae_lambda=gae_lambda,
        verbose=0,
        device='cpu'
    )

    env.enable_self_play(
        algorithm=PPO,
        save_path=None,
        agent=agent,
        opponent=opponent,
        model=model,
        update_interval=1,
        save_interval=0
    )

    check_env(env)

    results = []
    for i in range(100):  ################################################ 1e2
        print([i],end='')

        # print(env.game_board.agents, env.game_board.agents[0].model == env.game_board.agents[1].model)
        model.learn(total_timesteps=int(1e5))  ################################################ 1e5

        average_rewards, counts = evaluate_policy(
            model,
            env,
            n_eval_episodes=100,  ################################################ 1e3
            print_result=False,
            agents=[agent, random_agent],
            count_invalid_actions=0
        )

        results.append(average_rewards)

    average_rewards, counts = evaluate_policy(
        model,
        env,
        n_eval_episodes=10000,  ##############################################################1e4
        print_result=False,
        agents=[agent, random_agent],
        count_invalid_actions=0
    )

    print('%d hr %d min %d sec' % time_hr_min_sec(time.time() - start))

    save_name = '%.2f_' % average_rewards + save_name

    model.save(os.path.join(model_save_dir, save_name))

    plt.plot(np.arange(len(results)), np.array(results))
    plt.savefig(os.path.join(fig_save_dir, save_name + '.png'))
    plt.clf()
    plt.close()


def main():
    print('main')

    save_dir = 'hp_search/checkpoints/PPO'
    os.makedirs(save_dir, exist_ok=True)

    lr, invalid_action_reward, n_steps, cg_damping, n_critic_updates, gae_lambda = \
        0.03, -10, 10000, 0.1, 10, 0.95

    pool = ProcessPoolExecutor(max_workers=20)
    #pool.submit(learn, lr, invalid_action_reward, n_steps, cg_damping, n_critic_updates, gae_lambda)

    for lr_ in [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]:
        pool.submit(learn, lr_, invalid_action_reward, n_steps, cg_damping, n_critic_updates, gae_lambda)

    return

    for invalid_action_reward_ in [-100, -20, -5, -1]:
        pool.submit(learn, lr, invalid_action_reward_, n_steps, cg_damping, n_critic_updates, gae_lambda)

    for n_steps_ in [int(1e3), int(1e5)]:
        pool.submit(learn, lr, invalid_action_reward, n_steps_, cg_damping, n_critic_updates, gae_lambda)

    for cg_damping_ in [0.1, 0.01, 0.001]:
        pool.submit(learn, lr, invalid_action_reward, n_steps, cg_damping_, n_critic_updates, gae_lambda)

    for n_critics_updates_ in [5, 10, 15, 20]:
        pool.submit(learn, lr, invalid_action_reward, n_steps, cg_damping, n_critic_updates_, gae_lambda)

    for gae_lambda_ in [0.95, 0.98, 1.0]:
        pool.submit(learn, lr, invalid_action_reward, n_steps, cg_damping, n_critic_updates, gae_lambda_)



if __name__ == '__main__':
    main()