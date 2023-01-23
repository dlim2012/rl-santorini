
import gym
from gym import spaces
import numpy as np

from utils.self_play_wrapper import self_play_wrapper


@self_play_wrapper
class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, game_board):
        self.game_board = game_board
        self.action_space = spaces.Discrete(self.game_board.action_space_size)

        self.observation_space = spaces.MultiDiscrete(
            [self.game_board.observation_size] * self.game_board.observation_space_size
        )
        self.play = None

    def get_observation(self):
        return self.game_board.get_observation(self.game_board.learn_id)

    def reset(self):
        self.game_board.reset()
        self.play = self.game_board.play()
        return self.get_observation()

    def step(self, action):
        self.game_board.learn_agent.next_action = action
        return next(self.play)

    def render(self, mode='human'):
        self.game_board.render()

    def close(self):
        pass

    def get_valid_action_mask(self):
        return np.array(self.game_board.action_mask, dtype=np.uint8)





