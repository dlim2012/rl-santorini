
import gym
from gym import spaces
import numpy as np
import random
from itertools import chain

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
        for agent in game_board.agents:
            agent.action_space_type = spaces.Box

    def get_observation(self):
        return self.game_board.get_observation(self.game_board.learn_id)

    def reset(self):
        self.game_board.reset()
        #print('reset', self.get_observation())
        self.play = self.game_board.play()
        return self.get_observation()

    def step(self, action):
        self.game_board.learn_agent.next_action = action
        #res = next(self.play); print(res); return res;
        return next(self.play)

    def render(self, mode='human'):
        self.game_board.render(self.player_id)

    def close(self):
        pass

    def get_valid_action_mask(self):
        # just for masking
        return np.array(self.game_board.action_mask, dtype=np.uint8)





