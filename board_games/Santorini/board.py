"""
action:
    1) Choose one piece (male or female) -> action space 2
    2) Choose direction to move -> action space 8
    3) Choose direction to build -> action space 8

state:
    game board(25): 0~3(floor), 4(dome) for 25 slots
    locations(4n): (0~4(x-axis), 0~4(y-axis)) for 2 pieces per player
    next_move(1): 0(choose piece), 1(move), 2(build), 4(initialization stage)
    worker_type(1): 0(male), 1(female), 2(not selected)
    init_step(1): 0(male x-axis), 1(male y-axis), 2(female x-axis), 3(female y-axis), 4(initialize done)
    survive(n-1): survive list starting from the next turn
"""

import numpy as np
import random
from itertools import chain
from collections import defaultdict

from board_games.board_base import GameBoardBase


class GameBoard(GameBoardBase):
    def __init__(self, agents=None, learn_id=-1, invalid_action_reward=-10, print_simulation=False, mode=''):

        GameBoardBase.__init__(self, agents, learn_id, invalid_action_reward)

        self.print_simulation = print_simulation
        self.init_rand = 'init-rand' in mode
        self.print_occupied_spaces = True

        self.action_space_size = 8
        self.observation_size = 5
        self.observation_space_size = 27 + 5 * self.num_agents
        if self.init_rand:
            self.observation_space_size -= 1

        # observation variables
        self.buildings = None
        self.workers = None
        self.next_move = None
        self.worker_type = None
        self.init_move = None
        self.survive = None

        self.team_counts = None
        self.player_to_team = None
        self.winner_team = None
        self.rewards = None

        self.dx = (-1, 0, 1, 1, 1, 0, -1, -1)
        self.dy = (1, 1, 1, 0, -1, -1, -1, 0)

        self.action_mask = None
        self.occupied_locations = None

        self.n_turn, self.info = None, None
        self.turn = None

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
            self.rewards = {'win': 100, 'lose': -100 / (self.num_agents - 1),
                            'invalid_action': self.invalid_action_reward}
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
        self.worker_type = 2
        self.init_move = 0
        self.survive = [0] * self.num_agents

        # other variables
        self.team_counts = [0] * len(self.team_counts)

        self.occupied_locations = [[0] * 5 for x in range(5)]
        self.action_mask = self.init_valid_action_mask(0)

        self.n_turn = [0]
        self.info = defaultdict(list, {'n_turn': self.n_turn, 'invalid_action_count': 0})

        for agent in self.agents:
            agent.reset()
        self.turn = random.randint(0, self.num_agents - 1)

        if self.print_simulation:
            print('reset')

    def get_action(self, turn, agent, action_mask):
        obs = self.get_observation(turn)
        if turn == self.learn_id:
            yield -1, (obs, 0, False, self.info)
            action = agent.get_action(action_mask, obs)
            while action_mask[action] == 0:
                self.info['invalid_action_count'] += 1
                yield -1, (obs, self.rewards['invalid_action'], False, self.info)
                action = agent.get_action(action_mask, obs)
            yield action, None
        else:
            yield agent.get_action(action_mask, obs), None

    def play(self):
        if self.print_simulation:
            print('play', end=' '); self.render()

        # Initialize locations
        for i in range(self.num_agents):
            agent = self.agents[self.turn]
            for i in range(2):
                self.worker_type = i
                worker = self.workers[self.turn * 2 + i]

                # gather x-coordinate
                self.action_mask = self.init_valid_action_mask(0)
                if self.init_rand:
                    action = random.choice([i for i in range(5) if self.action_mask[i]])
                else:
                    for action, ret in self.get_action(self.turn, agent, self.action_mask):
                        if action == -1:
                            yield ret
                worker.x = action
                self.init_move += 1

                # gather y-coordinate
                self.action_mask = self.init_valid_action_mask(1, worker.x)
                if self.init_rand:
                    action = random.choice([i for i in range(5) if self.action_mask[i]])
                else:
                    for action, ret in self.get_action(self.turn, agent, self.action_mask):
                        if action == -1:
                            yield ret
                worker.y = action
                self.init_move += 1

                worker.survive = True
                self.occupied_locations[worker.x][worker.y] = 1
            self.survive[self.turn] = 1
            self.team_counts[self.player_to_team(self.turn)] += 1
            if self.print_simulation:
                self.render()

            self.init_move = 0
            self.n_turn[0] += 1
            self.turn = self.turn + 1 if self.turn != self.num_agents - 1 else 0
        else:
            self.init_move = 4
            self.next_move = 0

        # choose, move, and build
        while True:

            if self.survive[self.turn] == 0:
                continue

            available_workers, available_moves = self.get_available_moves(
                self.turn, self.buildings, self.occupied_locations, self.workers
            )
            if available_workers[0] == 0 and available_workers[1] == 0:
                self.team_counts[self.player_to_team(self.turn)] -= 1
                for worker_index in range(2*self.turn, 2*self.turn+2):
                    worker = self.workers[worker_index]
                    worker.survive = False
                    self.occupied_locations[worker.x][worker.y] = 0
                self.survive[self.turn] = 0
                self.info[self.n_turn[0]].append('player %d can\'t move' % self.turn)

                # Check if the game ended
                winner_team = self.check_survive_teams(self.team_counts)
                if winner_team != -1:
                    break

            agent = self.agents[self.turn]

            # choose worker
            self.next_move, self.worker_type = 0, 2
            self.action_mask = available_workers
            for action, ret in self.get_action(self.turn, agent, self.action_mask):
                if action == -1:
                    yield ret
            self.worker_type, worker = action, self.workers[2 * self.turn + action]

            # move the chosen worker
            self.next_move = 1
            self.action_mask = available_moves[self.worker_type]
            for action, ret in self.get_action(self.turn, agent, self.action_mask):
                if action == -1:
                    yield ret
            self.occupied_locations[worker.x][worker.y] = 0
            worker.x, worker.y = worker.x + self.dx[action], worker.y + self.dy[action]
            self.occupied_locations[worker.x][worker.y] = 1
            if self.buildings[worker.x][worker.y] == 3:
                self.info[self.n_turn[0]].append('%s went to level 3' % worker.name)
                winner_team = self.player_to_team(self.turn)
                break

            # build with the chosen worker
            self.next_move = 2
            self.action_mask = self.get_available_builds(worker.x, worker.y, self.occupied_locations)
            for action, ret in self.get_action(self.turn, agent, self.action_mask):
                if action == -1:
                    yield ret
            xx, yy = worker.x + self.dx[action], worker.y + self.dy[action]
            self.buildings[xx][yy] += 1
            if self.buildings[xx][yy] == 4:
                self.occupied_locations[xx][yy] = 1

            if self.print_simulation:
                self.render()

            self.n_turn[0] += 1
            self.turn = self.turn + 1 if self.turn != self.num_agents - 1 else 0

        if self.print_simulation:
            self.render()

        if self.learn_id >= 0:
            result = 'win' if self.player_to_team(self.learn_id) == winner_team else 'lose'
            self.info['result'] = result
            yield self.get_observation(self.learn_id), self.rewards[result], True, self.info
        else:
            yield winner_team

    def init_valid_action_mask(self, axis, x=None):
        if axis == 0:  # x-axis
            res = [1] * 5 + [0] * 3
            for x in range(5):
                if sum(self.occupied_locations[x]) == 5:
                    res[x] = 0
            return np.array(res)
        else:  # y-axis
            res = self.occupied_locations[x] + [1] * 3
            return 1 - np.array(res)

    def get_available_moves(self, turn, buildings, occupied_locations, workers):
        available_workers = [0] * self.action_space_size
        available_moves = [[0] * self.action_space_size for _ in range(2)]
        for worker_type in range(2):
            worker = workers[2 * turn + worker_type]
            for d in range(8):
                xx, yy = worker.x + self.dx[d], worker.y + self.dy[d]
                if xx < 0 or xx >= 5 or yy < 0 or yy >= 5:
                    continue
                if occupied_locations[xx][yy]:
                    continue
                if buildings[xx][yy] > buildings[worker.x][worker.y] + 1:
                    continue
                available_workers[worker_type] = 1
                available_moves[worker_type][d] = 1
        return available_workers, available_moves

    def get_available_builds(self, x, y, occupied_locations):
        res = [0] * self.action_space_size
        for d in range(8):
            xx, yy = x + self.dx[d], y + self.dy[d]
            if xx < 0 or xx >= 5 or yy < 0 or yy >= 5:
                continue
            if occupied_locations[xx][yy]:
                continue
            res[d] = 1
        return np.array(res)

    def check_survive_teams(self, team_counts):
        survive_team_count = sum([1 if count > 0 else 0 for count in team_counts])
        if survive_team_count == 1:
            for i, count in enumerate(team_counts):
                if count:
                    return self.player_to_team(i)
        else:
            return -1

    def render(self, print_line=True):
        locations = defaultdict(
            lambda: '__',
            {(worker.x, worker.y): worker.name for i, worker in enumerate(self.workers) if self.survive[i//2]}
        )
        if self.print_occupied_spaces:
            print({key: value for key, value in self.info.items()})
        for x in range(5):
            for y in range(5):
                print(self.buildings[x][y], end='  ')
            print(' | ', end='')
            for y in range(5):
                print(locations[(x, y)], end=' ')
            if self.print_occupied_spaces:
                print(' | ', end=' ')
                for y in range(5):
                    print(self.occupied_locations[x][y], end='  ')
            print()
        if print_line:
            print('-------------------------------------------------')

    def get_observation(self, player_id):
        # roll out workers and survive index as if the current player is player 0
        # assume that the current player is alive
        workers = self.workers[2*player_id:] + self.workers[:2*player_id]
        survive = self.survive[player_id+1:] + self.survive[:player_id]
        observation = \
            list(chain(*self.buildings)) + \
            list(chain(*[(worker.x, worker.y) for worker in workers])) + \
            survive + \
            [self.next_move] + \
            [self.worker_type]
        if not self.init_rand:
            observation += [self.init_move]
        return np.array(observation, dtype=np.uint8)


class Worker:
    def __init__(self, name=None, x=0, y=0, survive=False):
        self.name = name
        self.x = x
        self.y = y
        self.survive = survive


if __name__ == '__main__':
    from board_games.Santorini.agents import RandomAgent
    board = GameBoard([RandomAgent(), RandomAgent()], print_simulation=True)
    board.reset()
    play = board.play()
    next(play)
