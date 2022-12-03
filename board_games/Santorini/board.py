"""
action:
    1) Choose one piece (male or female) -> action space 2
    2) Choose direction to move -> action space 8
    3) Choose direction to build -> action space 8

state:
    game board(25): 0~3(floor), 4(dome) for 25 slots
    locations(4n): (0~4(x-axis), 0~4(y-axis)) for 2 pieces per player
    next_move(1): 0(choose piece), 1(move), 2(build), 4(initialization stage)
    worker_index(1): 0(not selected), 1(male), 2(female)
    init_step(1): 0(male x-axis), 1(male y-axis), 2(female x-axis), 3(female y-axis), 4(initialize done)
    survive(n-1): survive list starting from the next turn
"""

import numpy as np
import random
from itertools import chain

from collections import defaultdict


class GameBoard:

    def __init__(self, agents=None, learn_id=-1, invalid_action_reward=-10, print_simulation=False):


        self.num_agents = len(agents)
        self.agents = None
        self.learn_id, self.learn_agent = None, None
        self.invalid_action_reward = invalid_action_reward
        self.print_simulation = print_simulation

        self.action_space_size = 8
        self.observation_size = 5
        self.observation_space_size = 27 + 5 * self.num_agents

        # observation variables
        self.buildings = None
        self.workers = None
        self.next_move = None
        self.worker_index = None
        self.init_move = None
        self.survive = None # 1: True, 0: False

        self.available_workers = [0] * self.action_space_size
        self.available_moves = [[0] * self.action_space_size for _ in range(8)]

        self.team_counts = None
        self.player_to_team = None
        self.winner_team = None
        self.rewards = None

        self.dx = (-1, 0, 1, 1, 1, 0, -1, -1)
        self.dy = (1, 1, 1, 0, -1, -1, -1, 0)

        self.action_mask = None
        self.occupied_locations = None

        self.n_turn, self.info = None, None
        self.first_turn = None

        self.set_agents(agents=agents, learn_id=learn_id, invalid_action_reward=invalid_action_reward)

    def set_agents(self, agents=None, learn_id=None, invalid_action_reward=None):
        if agents != None:
            self.agents = agents
            for i, agent in enumerate(agents):
                agent.game_board, agent.agent_id = self, i
        if learn_id != None:
            self.learn_id, self.learn_agent = learn_id, agents[learn_id]
        if invalid_action_reward != None:
            self.invalid_action_reward = invalid_action_reward

        if self.num_agents in [2, 3]:
            self.team_counts = [1] * self.num_agents
            self.player_to_team = lambda x: x
            self.rewards = {'win': 100, 'lose': -100 / (self.num_agents - 1), 'invalid_action': self.invalid_action_reward}
        elif self.num_agents == 4:
            self.team_counts = [2] * 2
            self.player_to_team = lambda x: x % 2
            self.rewards = {'win': 100, 'lose': -100, 'invalid_action': self.invalid_action_reward}
        else:
            raise ValueError('Number of players %d is not in [2,3,4].' % self.num_agents)

    def reset(self):
        # observations
        self.buildings = [[0 for y in range(5)] for x in range(5)]
        self.workers = [Worker('%d%s' % (i, s)) for i in range(self.num_agents) for s in ['m', 'f']]
        self.next_move = 4
        self.worker_index = 0
        self.init_move = 0
        self.survive = [0] * self.num_agents

        # other variables
        self.team_counts = [0] * len(self.team_counts)

        self.occupied_locations = [[0] * 5 for x in range(5)]
        self.action_mask = self.init_valid_action_mask(0)

        self.n_turn = [0]
        self.info = defaultdict(list, {'n_turn': self.n_turn, 'invalid_action_count': 0})

        self.first_turn = random.randint(0, self.num_agents - 1)

        #print(self.agents)
        #print([agent.model for agent in self.agents], self.agents[0].model == self.agents[1].model)
        if self.print_simulation:
            print('reset')

    def get_action(self, turn, agent, action_mask):
        if turn == self.learn_id:
            obs = self.get_observation(turn)
            yield -1, (obs, 0, False, self.info)
            action = agent.get_action(action_mask)
            while action_mask[action] == 0:
                self.info['invalid_action_count'] += 1
                yield -1, (obs, self.rewards['invalid_action'], False, self.info)
                action = agent.get_action(action_mask)
            yield action, None
        else:
            yield agent.get_action(action_mask), None

    def play(self):
        turn = self.first_turn
        if self.print_simulation:
            print('play', end=' '); self.render()

        # Initialize locations
        for i in range(self.num_agents):
            agent = self.agents[turn]
            for i in range(2):
                self.worker_index = i
                self.worker_index = turn * 2 + i
                worker = self.workers[self.worker_index]

                # gather x-coordinate
                self.action_mask = self.init_valid_action_mask(0)
                action_gen = self.get_action(turn, agent, self.action_mask)
                action, ret = next(action_gen)
                while action == -1:
                    yield ret
                    action, ret = next(action_gen)
                worker.x = action
                self.init_move += 1

                # gather y-coordinate
                self.action_mask = self.init_valid_action_mask(1)
                action_gen = self.get_action(turn, agent, self.action_mask)
                action, ret = next(action_gen)
                while action == -1:
                    yield ret
                    action, ret = next(action_gen)
                worker.y = action
                self.init_move += 1

                worker.survive = True
                self.occupied_locations[worker.x][worker.y] = 1
            self.survive[turn] = 1
            self.team_counts[self.player_to_team(turn)] += 1
            if self.print_simulation:
                print('init', end=' '); self.render()

            self.init_move = 0
            self.n_turn[0] += 1
            turn = turn + 1 if turn != self.num_agents - 1 else 0
        else:
            self.init_move = 4
            self.next_move = 0

        # choose, move, and build
        while True:

            if self.survive[turn] == 0:
                continue

            if not self.set_available_moves(turn):
                self.team_counts[self.player_to_team(turn)] -= 1
                for worker_index in range(2*turn, 2*turn+2):
                    worker = self.workers[worker_index]
                    worker.survive = False
                    self.occupied_locations[worker.x][worker.y] = 0
                self.survive[turn] = 0
                self.info[self.n_turn[0]].append('player %d can\'t move' % turn)

                # Check if the game ended
                survive_team_count = sum([1 if count > 0 else 0 for count in self.team_counts])
                if survive_team_count == 1:
                    for i, count in enumerate(self.team_counts):
                        winner_team = self.player_to_team(i)
                        break
                    break
                continue

            agent = self.agents[turn]

            # choose worker
            self.next_move = 0
            self.action_mask = self.available_workers
            action_gen = self.get_action(turn, agent, self.action_mask)
            action, ret = next(action_gen)
            while action == -1:
                yield ret
                action, ret = next(action_gen)
            self.worker_index, worker = action, self.workers[action]

            # move the chosen worker
            self.next_move = 1
            self.action_mask = self.available_moves[self.worker_index]
            action_gen = self.get_action(turn, agent, self.action_mask)
            action, ret = next(action_gen)
            while action == -1:
                yield ret
                action, ret = next(action_gen)
            self.occupied_locations[worker.x][worker.y] = 0
            worker.x, worker.y = worker.x + self.dx[action], worker.y + self.dy[action]
            self.occupied_locations[worker.x][worker.y] = 1
            if self.buildings[worker.x][worker.y] == 3:
                self.info[self.n_turn[0]].append('%s went to level 3' % worker.name)
                winner_team = self.player_to_team(turn)
                break

            # build with the chosen worker
            self.next_move = 2
            self.action_mask = self.get_available_builds(worker)
            action_gen = self.get_action(turn, agent, self.action_mask)
            action, ret = next(action_gen)
            while action == -1:
                yield ret
                action, ret = next(action_gen)
            xx, yy = worker.x + self.dx[action], worker.y + self.dy[action]
            self.buildings[xx][yy] += 1
            if self.buildings[xx][yy] == 4:
                self.occupied_locations[xx][yy] = 1

            if self.print_simulation:
                print('move', end=' '); self.render()

            self.n_turn[0] += 1
            turn = turn + 1 if turn != self.num_agents - 1 else 0

        if self.print_simulation:
            self.render()
        if self.learn_id >= 0:
            result = 'win' if self.player_to_team(self.learn_id) == winner_team else 'lose'
            self.info['result'] = result
            yield self.get_observation(self.learn_id), self.rewards[result], True, self.info
        else:
            yield 0

    def init_valid_action_mask(self, axis):
        if axis == 0: # x-axis
            res = [1] * 5 + [0] * 3
            for x in range(5):
                if sum(self.occupied_locations[x]) == 5:
                    res[x] = 0
            return np.array(res)
        else: # y-axis
            x = self.workers[self.worker_index].x
            res = self.occupied_locations[x] + [1] * 3
            return 1 - np.array(res)

    def set_available_moves(self, turn):
        available = False
        self.available_workers = [0] * self.action_space_size
        for worker_index in range(2*turn, 2*turn+2):
            self.available_moves[worker_index] = [0] * self.action_space_size
            worker = self.workers[worker_index]
            for d in range(8):
                xx, yy = worker.x + self.dx[d], worker.y + self.dy[d]
                if xx < 0 or xx >= 5 or yy < 0 or yy >= 5:
                    continue
                if self.occupied_locations[xx][yy]:
                    continue
                if self.buildings[xx][yy] > self.buildings[worker.x][worker.y] + 1:
                    continue
                available = True
                self.available_moves[worker_index][d] = 1
                self.available_workers[worker_index] = 1
        return available

    def get_available_builds(self, worker):
        res = [0] * self.action_space_size
        x, y = worker.x, worker.y
        for d in range(8):
            xx, yy = x + self.dx[d], y + self.dy[d]
            if xx < 0 or xx >= 5 or yy < 0 or yy >= 5:
                continue
            if self.occupied_locations[xx][yy]:
                continue
            res[d] = 1
        return np.array(res)

    def render(self, print_line=True):
        locations = defaultdict(
            lambda: '__',
            {(worker.x, worker.y): worker.name for worker in self.workers if worker.survive}
        )
        print({key: value for key, value in locations.items()})
        print({key: value for key, value in self.info.items()})
        for x in range(5):
            for y in range(5):
                print(self.buildings[x][y], end='  ')
            print(' | ', end='')
            for y in range(5):
                print(locations[(x, y)], end=' ')
            print(' | ', end=' ')
            for y in range(5):
                print(self.occupied_locations[x][y], end='  ')
            print()
        if print_line:
            print('-------------------------------------------------')

    def get_observation(self, player_id):
        # roll out workers and survive index supposing that the current player is player 0
        # assume that the current player is alive
        workers = self.workers[2*player_id:] + self.workers[:2*player_id]
        survive = self.survive[player_id+1:] + self.survive[:player_id]
        worker_index = self.worker_index % 2
        observation = \
            list(chain(*self.buildings)) + \
            list(chain(*[(worker.x, worker.y) for worker in workers])) + \
            [self.next_move] + \
            [worker_index] + \
            [self.init_move] + \
            survive
        return np.array(observation, dtype=np.uint8)

class Worker:
    def __init__(self, name=None, x=0, y=0):
        self.name = name
        self.x = x
        self.y = y
        self.survive = False


if __name__ == '__main__':
    from board_games.Santorini.agents import RandomAgent
    board = GameBoard([RandomAgent(), RandomAgent()], print_simulation=True)
    board.reset()
    play = board.play()
    next(play)