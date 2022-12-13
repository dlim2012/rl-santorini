import gym
from gym import spaces
import numpy as np
import random


from utils.self_play_wrapper import self_play_wrapper

@self_play_wrapper
class TicTacToe(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, opponent=None):
        super().__init__()

        # declare spaces
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.MultiDiscrete([3 for x in range(3) for y in range(3)])

        # variables for game board and status
        self.game_board = None
        self.empty_count = None # number of empty elements in self.game_board
        self.player_id = 1 # Player ID
        self.opponent_id = 3 - self.player_id

        # if opponent is given as parameter, the opponent will play player 2
        self.opponent_mode = 'random' if opponent == None else 'agent'
        self.opponent = opponent

        # Reward function
        self.rewards = {'win': 100, 'lose': -100, 'tie': 0, 'default': 0}

        # Reset
        self.reset()

    def is_winner(self, x, y, player_id):

        # column match
        for xx in range(3):
            if self.game_board[xx][y] != player_id:
                break
        else:
            return True

        # row match
        for yy in range(3):
            if self.game_board[x][yy] != player_id:
                break
        else:
            return True

        # diagonal match (1)
        if x + y == 2:
            for xy in range(3):
                if self.game_board[2-xy][xy] != player_id:
                    break
            else:
                return True

        # diagonal match (2)
        if x - y == -2:
            for xy in range(3):
                if self.game_board[xy][xy] != player_id:
                    break
            else:
                return True

        return False

    def get_observation(self):
        return np.concatenate(self.game_board)

    def step(self, action):

        # Action
        x, y = divmod(action, 3)
        assert self.game_board[x][y] == 0, "(%d, %d)" % (x, y)

        # Next state
        self.game_board[x][y] = self.player_id
        self.empty_count -= 1
        if self.is_winner(x, y, self.player_id):
            return self.get_observation(), self.rewards['win'], True, {'done': 'Player %d won' % self.player_id, 'result': 'win'}
        if self.empty_count <= 0:
            return self.get_observation(), self.rewards['tie'], True, {'done': 'Tie', 'result': 'tie'}

        # Opponent action
        x, y = self.opponent_action()
        assert self.game_board[x][y] == 0

        # Opponent next state
        self.game_board[x][y] = 2
        self.empty_count -= 1
        if self.is_winner(x, y, 2):
            return self.get_observation(), self.rewards['lose'], True, {'done': 'Player %d won' % self.opponent_id, 'result': 'lose'}
        if self.empty_count <= 0:
            return self.get_observation(), self.rewards['tie'], True, {'done': 'Tie', 'result': 'tie'}

        return self.get_observation(), self.rewards['default'], False, {'print': True}

    def opponent_action(self):
        if self.opponent_mode == 'random':
            choices = [(x, y) for x in range(3) for y in range(3) if self.game_board[x][y] == 0]
            x, y = random.choice(choices)
        elif self.opponent_mode == 'agent':
            action, _states = self.opponent.predict(self.get_observation(), action_masks=self.get_valid_action_mask_opponent())
            x, y = divmod(action, 3)
        elif self.opponent_mode == 'human':
            # return human inputs
            self.render(print_line=False)
            while True:
                print('Enter row number and column number:')
                try:
                    x = int(input())
                    y = int(input())
                except:
                    print('Invalid input. Enter two integers, one in each line.')
                    continue
                if 0 <= x < 3 and 0 <= y < 3 and self.game_board[x][y] == 0:
                    break
                else:
                    print("Invalid input: (%d, %d). Enter integers between 0, 1, and 2 indicating an empty slot.")
        else:
            raise ValueError()
        return x, y

    def reset(self):
        self.game_board = np.array([[0 for y in range(3)] for x in range(3)], dtype=np.uint8)
        self.empty_count = 9

        # With 50% probability start with player 2
        if random.random() < 0.5:
            x, y = self.opponent_action()
            self.game_board[x][y] = 2
            self.empty_count -= 1

        return self.get_observation()

    def render(self, mode='human', print_line=True):
        for x in range(3):
            for y in range(3):
                print(self.game_board[x][y], end=' ')
            print()
        if print_line:
            print('---------------------')

    def close(self):
        pass

    def get_valid_action_mask(self):
        res = [1 if self.game_board[x][y] == 0 else 0 for x in range(3) for y in range(3)]
        return np.array(res)

    def get_valid_action_mask_opponent(self):
        res = [1 if self.game_board[x][y] == 0 else 0 for x in range(3) for y in range(3)]
        return np.array(res)

