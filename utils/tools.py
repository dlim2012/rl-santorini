
from stable_baselines3.common.utils import obs_as_tensor
import numpy as np
import random

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def predict_with_mask(model, obs, game_board):
    obs = obs_as_tensor(np.array([obs], dtype=np.int64), model.policy.device)

    x = model.policy.get_distribution(obs).distribution.probs.detach().cpu().numpy()[0]

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


