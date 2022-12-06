
from stable_baselines3.common.utils import obs_as_tensor
import torch as th
import numpy as np
import random

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def predict_with_mask(model, obs, game_board, mode='PPO'):
    obs = obs_as_tensor(np.array([obs], dtype=np.int64), model.policy.device)

    if mode in ['PPO', 'A2C', 'TRPO']:
        x = model.policy.get_distribution(obs).distribution.probs.detach().numpy()[0]
    elif mode == 'DQN':
        x = th.exp(model.policy.q_net(obs)[0])

    choices, weights = [], []
    for i in range(game_board.action_space_size):
        if game_board.action_mask[i] != 0:
            choices.append(i)
            weights.append(x[i])
    action = random.choices(choices, weights=weights)[0]
    return action

def time_hr_min_sec(t):
    hr = t // 3600
    t -= hr * 3600
    min = t // 60
    t -= min * 60
    sec = int(t)
    return (hr, min, sec)
####################################################################################

""" Archive
def masked():
    print('masked')

    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.ppo_mask import MaskablePPO

    agent = RLAgent(learning=True)
    n_steps = int(1e5)

    game_board = GameBoard([agent, RandomAgent()], learn_id=0)
    env = Santorini(game_board)
    masked_env = ActionMasker(env, mask_fn)  # Wrap to enable masking

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        masked_env,
        learning_rate=0.03,
        n_steps=n_steps,
        verbose=1)

    for i in range(int(1e4)):
        try:
            model.learn(total_timesteps=int(1e4))
        except:
            print(env.get_observation())
"""