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
from utils.tools import time_hr_min_sec
from concurrent.futures import ProcessPoolExecutor

def learn(lr, invalid_action_reward, n_steps, gae_lambda, ent_coef, vf_coef, use_rms_prop):
    start = time.time()

    model_save_dir = 'hp_search/checkpoints/PPO'
    fig_save_dir = 'hp_search/results/PPO'
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(fig_save_dir, exist_ok=True)

    save_name = 'lr%.2e_ir%.2e_ns%.2e_gl%.2e_ec%.2e_fc%.2e_rp%d' % \
                (lr, invalid_action_reward, n_steps, gae_lambda, ent_coef, vf_coef, 1 if use_rms_prop else 0)
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

    model = A2C(
        'MlpPolicy',
        env,
        learning_rate=lr,
        n_steps=n_steps,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        use_rms_prop=use_rms_prop,
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
        save_interval=0,
    )

    check_env(env)

    results = []
    for i in range(100):  ################################################ 1e2
        # print(env.game_board.agents, env.game_board.agents[0].model == env.game_board.agents[1].model)
        print([i],end='')

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

    lr, invalid_action_reward, n_steps, gae_lambda, ent_coef, vf_coef, use_rms_prop = \
        0.03, -10, 10000, 1.0, 0.0, 0.5, True

    pool = ProcessPoolExecutor(max_workers=16)
    #pool.submit(learn, lr, invalid_action_reward, n_steps, gae_lambda, ent_coef, vf_coef, use_rms_prop)

    for lr_ in [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]:
        for use_rms_prop_ in [True, False]:
            pool.submit(learn, lr_, invalid_action_reward, n_steps, gae_lambda, ent_coef, vf_coef, use_rms_prop_)

    return

    for invalid_action_reward_ in [-100, -20, -5, -1]:
            pool.submit(learn, learn, lr, invalid_action_reward_, n_steps, gae_lambda, ent_coef, vf_coef, use_rms_prop)

    for n_steps_ in [int(1e3), int(1e5)]:
            pool.submit(learn, learn, lr, invalid_action_reward, n_steps_, gae_lambda, ent_coef, vf_coef, use_rms_prop)

    for gae_lambda_ in [0.8, 0.9, 0.95]:
            pool.submit(learn, learn, lr, invalid_action_reward, n_steps, gae_lambda_, ent_coef, vf_coef, use_rms_prop)

    for ent_coef_ in [0.001, 0.01]:
            pool.submit(learn, learn, lr, invalid_action_reward, n_steps, gae_lambda, ent_coef_, vf_coef, use_rms_prop)

    for vf_coef_ in [0.2, 0.3, 0.4]:
            pool.submit(learn, learn, lr, invalid_action_reward, n_steps, gae_lambda, ent_coef, vf_coef_, use_rms_prop)


if __name__ == '__main__':
    main()