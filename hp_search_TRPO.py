from stable_baselines3 import A2C, DQN, PPO
from sb3_contrib import TRPO, ARS
import os
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from argparse import ArgumentParser
from tensorboardX import SummaryWriter

from board_games.Santorini.agents import RandomAgent, RLAgent, MiniMaxAgent
from board_games.Santorini.board import GameBoard
from board_games.CustomEnv import CustomEnv
from utils.evaluate import evaluate_policy
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count


def learn(lr, invalid_action_reward, n_steps, cg_damping, n_critic_updates, gae_lambda):
    total_timesteps = int(2e7) ############ 2e7

    n_iter = total_timesteps // n_steps
    eval_period = int(1e5) // n_steps ############### int(1e5)
    if eval_period == 0:
        eval_period = 1

    model_save_dir = 'hp_search/checkpoints/PPO'
    fig_save_dir = 'hp_search/results/PPO'
    log_dir = 'logs/PPO'
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(fig_save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    save_name = 'lr%.2e_ir%.2e_ns%.2e_cd%.2e_cu%.2e_gl%.2e' % \
                (lr, invalid_action_reward, n_steps, cg_damping, n_critic_updates, gae_lambda)
    print(save_name)

    writer = SummaryWriter(os.path.join(log_dir, save_name))

    agent = RLAgent(learning=True)
    opponent = RLAgent(learning=False)
    opponents = [
        ('random', RandomAgent(), 100), ############################ 100
        ('minimax1', MiniMaxAgent(maxDepth=1), 100), ############################ 100
        ('minimax2', MiniMaxAgent(maxDepth=2), 20), ############################ 20
        ('minimax3', MiniMaxAgent(maxDepth=3), 4), ############################ 4
    ]

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

    for i in range(n_iter):
        print([i],end='')

        model.learn(total_timesteps=n_steps)

        if (i+1) % eval_period == 0:
            for name, opponent, n_eval_episodes in opponents:
                average_rewards, counts = evaluate_policy(
                    model,
                    env,
                    n_eval_episodes=n_eval_episodes,
                    print_result=False,
                    agents=[agent, opponent],
                    count_invalid_actions=int(1e3)
                )

                writer.add_scalar(name, average_rewards, n_steps * (i+1))
                writer.add_scalar('invalid_action_count_%s' % name, counts['invalid_action_count'], n_steps * (i+1))


    model.save(os.path.join(model_save_dir, save_name))

    for name, opponent, n_eval_episodes in opponents:
        average_rewards, counts = evaluate_policy(
            model,
            env,
            n_eval_episodes=n_eval_episodes * 100, ######################
            print_result=False,
            agents=[agent, opponent],
            count_invalid_actions=int(1e3)
        )
        writer.add_scalar(name + '_final', average_rewards, 1)

    writer.close()


def main():
    print('main')

    save_dir = 'hp_search/checkpoints/PPO'
    os.makedirs(save_dir, exist_ok=True)

    lr, invalid_action_reward, n_steps, cg_damping, n_critic_updates, gae_lambda = \
        0.03, -10, 10000, 0.1, 10, 0.95

    pool = ProcessPoolExecutor(max_workers=cpu_count())
    #learn(lr, invalid_action_reward, n_steps, cg_damping, n_critic_updates, gae_lambda)


    for lr_ in [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]:
        pool.submit(learn, lr_, invalid_action_reward, n_steps, cg_damping, n_critic_updates, gae_lambda)

    return
    """
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
    """

if __name__ == '__main__':
    main()