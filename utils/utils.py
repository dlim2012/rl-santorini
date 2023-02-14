from board_games.Santorini.agents import HumanAgent, RLAgent, MiniMaxAgent, RandomAgent

from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO


def get_agent(agent_type, algorithm=None, ckpt_path=None):
    if agent_type == 'random':
        agent = RandomAgent()
    elif agent_type[:7] == 'minimax':
        assert len(agent_type) == 8 and agent_type[7].isdecimal()
        agent = MiniMaxAgent(max_depth=int(agent_type[7]))
    elif agent_type == 'rl':
        agent = RLAgent()
        if algorithm == 'TRPO':
            agent.model = TRPO.load(ckpt_path)
        elif algorithm == 'PPO':
            agent.model = PPO.load(ckpt_path)
        elif algorithm == 'A2C':
            agent.model = A2C.load(ckpt_path)
        else:
            raise ValueError()
    elif agent_type == 'human':
        agent = HumanAgent()
    elif agent_type == 'none':
        agent = 'none'
    else:
        raise ValueError('%s is not available for opponent.')
    return agent