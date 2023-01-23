from .agents import RandomAgent, MiniMaxAgent


def get_eval_opponents(mode):
    if mode == 'default':
        eval_opponents = [
            ('random', RandomAgent(), 100),
            ('minimax1', MiniMaxAgent(max_depth=1), 100),
            ('minimax2', MiniMaxAgent(max_depth=2), 20),
            ('minimax3', MiniMaxAgent(max_depth=3), 10),
        ]
    else:
        raise ValueError()

    return eval_opponents
