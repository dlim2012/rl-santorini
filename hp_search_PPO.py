from stable_baselines3 import A2C, DQN, PPO
from sb3_contrib import TRPO, ARS
import os
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from tensorboardX import SummaryWriter

from board_games.Santorini.agents import RandomAgent, RLAgent, MiniMaxAgent
from board_games.Santorini.board import GameBoard
from board_games.CustomEnv import CustomEnv
from utils.evaluate import evaluate_policy
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count


def learn(lr, invalid_action_reward, n_steps, gae_lambda, n_epochs, clip_range, ent_coef, vf_coef, target_kl):

    total_timesteps = int(2e7)  ############ 2e7

    n_iter = total_timesteps // n_steps
    eval_period = int(1e5) // n_steps  ############### int(1e5)
    if eval_period == 0:
        eval_period = 1

    model_save_dir = 'hp_search/checkpoints/PPO'
    fig_save_dir = 'hp_search/results/PPO'
    log_dir = 'logs/PPO'
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(fig_save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    save_name = 'lr%.2e_ir%.2e_ns%.2e_ne%.2e_cr%.2e_ent%.2e_tkl%.2e' % \
                (lr, invalid_action_reward, n_steps, n_epochs, clip_range, ent_coef, target_kl)
    print(save_name)

    writer = SummaryWriter(os.path.join(log_dir, save_name))

    agent = RLAgent(learning=True)
    opponent = RLAgent(learning=False)
    opponents = [
        ('random', RandomAgent(), 100),  ############################ 100
        ('minimax1', MiniMaxAgent(maxDepth=1), 100),  ############################ 100
        ('minimax2', MiniMaxAgent(maxDepth=2), 20),  ############################ 20
        ('minimax3', MiniMaxAgent(maxDepth=3), 4),  ############################ 4
    ]

    game_board = GameBoard(
        agents=[agent, opponent],
        learn_id=0,
        invalid_action_reward=invalid_action_reward
    )
    env = CustomEnv(game_board)
    env = Monitor(env)

    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=lr,
        n_steps=n_steps,
        gae_lambda=gae_lambda,
        n_epochs=n_epochs,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        target_kl=target_kl,
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
        print([i], end='')

        model.learn(total_timesteps=n_steps)

        if (i + 1) % eval_period == 0:
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

            model.save(os.path.join(model_save_dir, save_name))

    for name, opponent, n_eval_episodes in opponents:
        average_rewards, counts = evaluate_policy(
            model,
            env,
            n_eval_episodes=n_eval_episodes * 100,  ######################
            print_result=False,
            agents=[agent, opponent],
            count_invalid_actions=int(1e3)
        )
        writer.add_scalar(name + '_final', average_rewards, 1)

    writer.close()


def main():
    print('main')

    lr, invalid_action_reward, n_steps, gae_lambda, n_epochs, clip_range, ent_coef, vf_coef, target_kl = \
        0.03, -10, 10000, 1.0, 10, 0.2, 0.0, 0.5, 0.01,

    pool = ProcessPoolExecutor(max_workers=cpu_count())
    #pool.submit(learn, lr, invalid_action_reward, n_steps, gae_lambda, n_epochs, clip_range, ent_coef, vf_coef, target_kl)

    #for lr_ in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
    #    pool.submit(learn, lr_, invalid_action_reward, n_steps, gae_lambda, n_epochs, clip_range, ent_coef, vf_coef, target_kl)

    #for lr_ in [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]:
    for lr_ in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
        pool.submit(learn, lr_, invalid_action_reward, n_steps, gae_lambda, n_epochs, clip_range, ent_coef, vf_coef, target_kl)

    return

    for invalid_action_reward_ in [-100, -20, -5, -1]:
        pool.submit(learn, lr, invalid_action_reward_, n_steps, gae_lambda, n_epochs, clip_range, ent_coef, vf_coef, target_kl)

    for n_steps_ in [int(1e3), int(1e5)]:
        pool.submit(learn, lr, invalid_action_reward, n_steps_, gae_lambda, n_epochs, clip_range, ent_coef, vf_coef, target_kl)

    for gae_lambda_ in [0.8, 0.9, 0.95]:
        pool.submit(learn, lr, invalid_action_reward, n_steps, gae_lambda_, n_epochs, clip_range, ent_coef, vf_coef, target_kl)

    for n_epochs_ in [3, 30]:
        pool.submit(learn, lr, invalid_action_reward, n_steps, gae_lambda, n_epochs_, clip_range, ent_coef, vf_coef, target_kl)

    for clip_range_ in [0.1, 0.3]:
        pool.submit(learn, lr, lr, invalid_action_reward, n_steps, gae_lambda, n_epochs, clip_range_, ent_coef, vf_coef, target_kl)

    for ent_coef_ in [0.01]:
        pool.submit(learn, lr, invalid_action_reward, n_steps, gae_lambda, n_epochs, clip_range, ent_coef_, vf_coef, target_kl)

    for vf_coef_ in [0.2, 0.3, 0.4]:
        pool.submit(learn, lr, invalid_action_reward, n_steps, gae_lambda, n_epochs, clip_range, ent_coef, vf_coef_, target_kl)

    for target_kl_ in [0.003, 0.03]:
        pool.submit(learn, lr, invalid_action_reward, n_steps, gae_lambda, n_epochs, clip_range, ent_coef, vf_coef, target_kl_)

if __name__ == '__main__':
    main()