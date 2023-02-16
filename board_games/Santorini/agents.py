

import random
from utils.tools import predict_with_mask
from collections import deque

from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO


from .board import Worker

class AgentBase:

    def __init__(self, agent_type):
        self.agent_type = agent_type

        self.game_board = None
        self.agent_id = None

        self.model = None
        self.require_obs = False
        self.use_gui = False

    def get_action(self, action_mask=None, obs=None):
        raise NotImplementedError()

    def reset(self):
        return


class RandomAgent(AgentBase):

    def __init__(self):
        super(RandomAgent, self).__init__("random")

    def get_action(self, action_mask=None, obs=None):
        choices = [i for i in range(self.game_board.action_space_size) if int(action_mask[i]) == 1]
        return random.choice(choices)


class HumanAgent(AgentBase):

    def __init__(self):
        super(HumanAgent, self).__init__("human")
        self.next_actions = [-1, -1]

    def get_action(self, action_mask=None, obs=None):
        """ Command line user interface """
        #print('Directions: clockwise (upper right(0), right(1), ..., up(7))')
        text = '[Turn %d] ' % self.game_board.turn
        if self.game_board.init_move < 4:
            axis = 'x-axis' if self.game_board.init_move & 1 == 0 else 'y-axis'
            worker_type = 'male' if self.game_board.init_move < 2 else 'female'
            text += 'Choose the {axis} for the {worker_type} worker.'.format(axis=axis, worker_type=worker_type)
        elif self.game_board.next_move == 0:
            text += 'Choose worker.'
        elif self.game_board.next_move == 1:
            text += 'Choose a direction to move.'
        else:
            text += 'Choose a direction to build.'

        while True:
            try:
                action = int(input(text + ' (action mask:' + str(action_mask) + '): '))
            except:
                print('Invalid input, enter an integer.')
                continue
            if action_mask[action] == 0:
                print('Invalid action.')
            else:
                return action


class RLAgent(AgentBase):
    def __init__(self, learning=False):
        super(RLAgent, self).__init__("rl")
        self.model = None
        self.next_action = None
        self.learning = learning
        self.require_obs = True

    def get_action(self, action_mask=None, obs=None):
        if self.learning:
            action = self.next_action
        else:
            action = predict_with_mask(self.model, obs, self.game_board)
        return action

    def set_model(self, algorithm, ckpt_path):
        self.model = algorithms[algorithm].load(ckpt_path)

class MiniMaxAgent(AgentBase):
    def __init__(self, max_depth):
        super(MiniMaxAgent, self).__init__("minimax")
        self.next_actions = deque()
        self.maxDepth = max_depth

    def get_action(self, action_mask=None, obs=None):
        if self.game_board.next_move != 0:
            if self.game_board.init_move < 4:
                choices = [i for i in range(self.game_board.action_space_size) if int(action_mask[i]) == 1]
                action = random.choice(choices)
                return action
            else:
                return self.next_actions.popleft()
        else:
            # state: turn, buildings, occupied_locations, workers, survive, team_counts
            state = (
                self.agent_id,
                self.game_board.buildings,
                self.game_board.occupied_locations,
                self.game_board.workers,
                self.game_board.survive,
                self.game_board.team_counts
            )

            maxScore, actions = self.minimax(0, state, True)
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

        available_workers, available_moves = self.game_board.get_available_moves(
            turn, buildings, occupied_locations, workers
        )

        # no possible moves for both workers
        if available_workers[0] == 0 and available_workers[1] == 0:
            new_survive, new_team_counts = survive[:], team_counts[:]
            new_survive[turn] = 0
            new_team_counts[self.game_board.player_to_team(turn)] -= 1
            next_turn = self.get_next_turn(turn, new_survive)
            next_state = (next_turn, buildings, occupied_locations, workers, new_survive, team_counts)
            cont_states.append((next_state, None))
            return cont_states, 0

        for worker_type in range(2):
            # no possible moves
            if not available_workers[worker_type]:
                continue

            worker_index = 2 * turn + worker_type
            worker = workers[worker_index]

            # move
            for action1, valid1 in enumerate(available_moves[worker_type]):
                if not valid1:
                    continue

                new_occupied_locations = occupied_locations[:]
                new_workers = workers[:]

                new_occupied_locations[worker.x] = occupied_locations[worker.x][:]
                new_occupied_locations[worker.x][worker.y] = 0
                new_worker = Worker(worker.name, worker.x + self.game_board.dx[action1], worker.y + self.game_board.dy[action1])
                if new_worker.x != worker.x:
                    new_occupied_locations[new_worker.x] = occupied_locations[new_worker.x][:]
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
                for action2, valid2 in enumerate(available_builds):
                    if not valid2:
                        continue
                    new_buildings = buildings[:]
                    xx, yy = new_worker.x + self.game_board.dx[action2], new_worker.y + self.game_board.dy[action2]
                    new_buildings[xx] = buildings[xx][:]
                    new_buildings[xx][yy] += 1
                    next_turn = self.get_next_turn(turn, survive)
                    if new_buildings[xx][yy] == 4:
                        new_new_occupied_locations = new_occupied_locations[:]
                        new_new_occupied_locations[xx] = new_occupied_locations[xx][:]
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

        bestActionsList = []
        if maxTurn:
            maxScore = -float('inf')
            for next_state, next_actions in next_states:
                score, actions = self.minimax(depth+1, next_state, False)
                if score > maxScore:
                    maxScore, bestActionsList = score, [next_actions]
                elif score == maxScore:
                    bestActionsList.append(next_actions)
            return maxScore, random.choice(bestActionsList)
        else:
            minScore = float('inf')
            for next_state, next_actions in next_states:
                score, actions = self.minimax(depth+1, next_state, True)
                if score < minScore:
                    minScore, bestActionsList = score, [next_actions]
                elif score == minScore:
                    bestActionsList.append(next_actions)
            return minScore, random.choice(bestActionsList)

    def render(self, state, print_line=True):
        from collections import defaultdict
        turn, buildings, occupied_locations, workers, survive, team_counts = state

        locations = defaultdict(
            lambda: '__',
            {(worker.x, worker.y): worker.name for i, worker in enumerate(workers) if survive[i // 2]}
        )
        for x in range(5):
            for y in range(5):
                print(buildings[x][y], end='  ')
            print(' | ', end='')
            for y in range(5):
                print(locations[(x, y)], end=' ')
            print(' | ', end=' ')
            for y in range(5):
                print(occupied_locations[x][y], end='  ')
            print()
        if print_line:
            print('-------------------------------------------------')

    def reset(self):
        self.next_actions = deque()

algorithms = {
    'TRPO': TRPO,
    'PPO': PPO,
    'A2C': A2C
}

def get_agent(agent_type, algorithm=None, ckpt_path=None):
    if agent_type == 'random':
        agent = RandomAgent()
    elif agent_type[:7] == 'minimax':
        assert len(agent_type) == 8 and agent_type[7].isdecimal()
        agent = MiniMaxAgent(max_depth=int(agent_type[7]))
    elif agent_type == 'rl':
        agent = RLAgent()
        if ckpt_path:
            agent.model = algorithms[algorithm].load(ckpt_path)
    elif agent_type == 'human':
        agent = HumanAgent()
    elif agent_type == 'none':
        agent = 'none'
    else:
        raise ValueError('%s is not available for opponent.' % agent_type)
    return agent