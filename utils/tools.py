import sys
import random

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

####################################################################################

""" Archive
def masked():
    print('masked')

    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.ppo_mask import MaskablePPO

    agent = RLAgent(learning=True)
    n_steps = int(1e5)

    game_board = GameBoard([agent, RandomAgent()], learn_id=0)
    env = Santorini(game_board)
    masked_env = ActionMasker(env, mask_fn)  # Wrap to enable masking

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        masked_env,
        learning_rate=0.03,
        n_steps=n_steps,
        verbose=1)

    for i in range(int(1e4)):
        try:
            model.learn(total_timesteps=int(1e4))
        except:
            print(env.get_observation())
"""