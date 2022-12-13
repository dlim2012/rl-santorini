
import numpy as np
import random
from itertools import chain

from collections import defaultdict

class GameBoardBase:
    def __init__(self, agents=None, learn_id=-1, invalid_action_reward=-10):

        self.num_agents = len(agents)
        self.agents = None
        self.learn_id, self.learn_agent = None, None

        self.action_space_size = None
        self.observation_size = None
        self.observation_space_size = None


    def set_agents(self, agents=None, learn_id=None, invalid_action_reward=None):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError

    def play(self):
        raise NotImplementedError()

    def render(self, print_line=True):
        raise NotImplementedError()

    def get_observation(self, player_id):
        raise NotImplementedError()
