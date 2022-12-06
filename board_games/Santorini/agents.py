
import random
import numpy as np
from utils.tools import predict_with_mask
from collections import deque
import gym
from board_games.Santorini.board import Worker

class AgentBase:

    def __init__(self):
        self.game_board = None
        self.agent_id = None
        self.model = None

    def get_action(self, action_mask, obs=None):
        raise NotImplementedError()

    def reset(self):
        pass

class RandomAgent(AgentBase):

    def __init__(self):
        super(RandomAgent, self).__init__()

    def get_action(self, action_mask, obs=None):
        choices = [i for i in range(self.game_board.action_space_size) if int(action_mask[i]) == 1]
        return random.choice(choices)


class RLAgent(AgentBase):
    def __init__(self, learning=False):
        super(RLAgent, self).__init__()
        self.model = None
        self.next_action = None
        self.learning = learning

    def get_action(self, action_mask, obs=None):
        if self.learning:
            action = self.next_action
        else:
            action = predict_with_mask(self.model, obs, self.game_board)
        return action

class MiniMaxAgent(AgentBase):
    def __init__(self, maxDepth):
        super(MiniMaxAgent, self).__init__()
        self.next_actions = deque()
        self.maxDepth = maxDepth

    def get_action(self, action_mask, obs=None):
        if self.game_board.next_move != 0:
            if self.game_board.init_move < 4:
                choices = [i for i in range(self.game_board.action_space_size) if int(action_mask[i]) == 1]
                action = random.choice(choices)
                return action
            else:
                return self.next_actions.popleft()
        else:
            turn = self.agent_id
            buildings = [row[:] for row in self.game_board.buildings]
            occupied_locations = [row[:] for row in self.game_board.occupied_locations]
            workers = [Worker(worker.name, worker.x, worker.y) for worker in self.game_board.workers]
            survive = self.game_board.survive[:]
            team_counts = self.game_board.team_counts[:]
            state = (turn, buildings, occupied_locations, workers, survive, team_counts)
            maxScore, actions = self.minimax(0, state, True)
            #print(maxScore, actions)
            self.next_actions.extend(actions[1:])
            return actions[0]

    def get_next_turn(self, turn, survive):
        turn += 1
        if turn == self.game_board.num_agents:
            turn -= self.game_board.num_agents
        while survive[turn] == 0:
            turn += 1
            if turn == self.game_board.num_agents:
                turn -= self.game_board.num_agents
        return turn

    def gather_next_states(self, state, maxTurn):
        win_states, cont_states, lose_states = [], [], []
        turn, buildings, occupied_locations, workers, survive, team_counts = state

        # no possible moves for both workers
        if not self.game_board.set_available_moves(turn, buildings, occupied_locations, workers):
            new_survive, new_team_counts = survive[:], team_counts[:]
            new_survive[turn] = 0
            team_counts[self.game_board.player_to_team(turn)] -= 1
            next_turn = self.get_next_turn(turn, new_survive)
            next_state = (next_turn, buildings, occupied_locations, workers, new_survive, team_counts)
            cont_states.append((next_state, None))
            return cont_states, 0

        for worker_type in range(2):
            # no possible moves
            if not self.game_board.available_workers[worker_type]:
                continue

            worker_index = 2 * turn + worker_type
            worker = workers[worker_index]

            # move
            for action1, valid in enumerate(self.game_board.available_moves[worker_type]):
                if not valid:
                    continue

                new_occupied_locations = [row[:] for row in occupied_locations]
                new_workers = workers[:]

                new_occupied_locations[worker.x][worker.y] = 0
                new_worker = Worker(worker.name, worker.x + self.game_board.dx[action1], worker.y + self.game_board.dy[action1])
                new_occupied_locations[new_worker.x][new_worker.y] = 1
                new_workers[worker_index] = new_worker

                if buildings[new_worker.x][new_worker.y] == 3:
                    winner_team = self.game_board.player_to_team(turn)
                    next_turn = self.get_next_turn(turn, survive)
                    next_state = (next_turn, buildings, new_occupied_locations, new_workers, survive, team_counts)
                    if self.game_board.player_to_team(self.agent_id) == winner_team:
                        win_states.append((next_state, (worker_type, action1, -1)))
                        if maxTurn:
                            return win_states, 1
                    else:
                        lose_states.append((next_state, (worker_type, action1, -1)))
                        if not maxTurn:
                            return lose_states, -1
                # build
                available_builds = self.game_board.get_available_builds(new_worker.x, new_worker.y, new_occupied_locations)
                for action2, valid in enumerate(available_builds):
                    new_buildings = [row[:] for row in buildings]
                    if not valid:
                        continue
                    xx, yy = new_worker.x + self.game_board.dx[action2], new_worker.y + self.game_board.dy[action2]
                    new_buildings[xx][yy] += 1
                    next_turn = self.get_next_turn(turn, survive)
                    if new_buildings[xx][yy] == 4:
                        new_new_occupied_locations = [row[:] for row in new_occupied_locations]
                        new_new_occupied_locations[xx][yy] = 1
                    else:
                        new_new_occupied_locations = new_occupied_locations
                    next_state = (next_turn, new_buildings, new_new_occupied_locations, new_workers, survive, team_counts)
                    cont_states.append((next_state, (worker_type, action1, action2)))

        if len(cont_states) != 0:
            return cont_states, 0
        return (lose_states, -1) if maxTurn else (win_states, 1)


    def minimax(self, depth, state, maxTurn):

        if depth > 0:
            # check end game
            winner_team = self.game_board.check_survive_teams(state[-1])
            if winner_team != -1:
                if self.game_board.player_to_team(self.agent_id) == winner_team:
                    return self.game_board.rewards['win'], None
                else:
                    return self.game_board.rewards['lose'], None

        # max depth reached
        if depth == self.maxDepth:
            return 0, None

        next_states, result = self.gather_next_states(state, maxTurn)

        if result == 1:
            return self.game_board.rewards['win'], random.choice(next_states)[1]
        elif result == -1:
            return self.game_board.rewards['lose'], random.choice(next_states)[1]

        if maxTurn:
            maxScore, bestActions = -float('inf'), None
            for next_state, next_actions in next_states:
                score, actions = self.minimax(depth+1, next_state, False)
                if score > maxScore:
                    maxScore, bestActions = score, next_actions
            return maxScore, bestActions
        else:
            minScore, bestActions = float('inf'), None
            for next_state, next_actions in next_states:
                score, actions = self.minimax(depth+1, next_state, True)
                if score < minScore:
                    minScore, bestActions = score, next_actions
            return minScore, bestActions

    def reset(self):
        self.next_actions = deque()