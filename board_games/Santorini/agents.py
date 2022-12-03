
import random
import numpy as np
from utils.evaluate import predict_with_mask
import gym

class AgentBase:

    def __init__(self):
        self.game_board = None
        self.agent_id = None
        self.model = None

    def get_action(self, action_mask):
        raise NotImplementedError()

class RandomAgent(AgentBase):

    def __init__(self):
        super(RandomAgent, self).__init__()

    def get_action(self, action_mask):
        choices = [i for i in range(self.game_board.action_space_size) if int(action_mask[i]) == 1]
        if len(choices) == 0:
            return -1
        return random.choice(choices)

class HumanAgent(AgentBase):

    def __init__(self):
        super(HumanAgent, self).__init__()

    def get_action(self, action_mask):
        choices = [i for i in range(self.game_board.action_space_size) if int(action_mask[i]) == 1]
        if len(choices) == 0:
            return -1
        return random.choice(choices)

class RLAgent(AgentBase):
    def __init__(self, learning=False):
        super(RLAgent, self).__init__()
        self.model = None
        self.next_action = None
        self.learning = learning

    def get_action(self, action_mask):
        if self.learning:
            action = self.next_action
        else:
            obs = self.game_board.get_observation(self.agent_id)
            action = predict_with_mask(self.model, obs, self.game_board)
        return action

