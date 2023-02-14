import random

def self_play_wrapper(cls):

    class SelfPlayWrapper(cls):
        """
        Supports two agent games only (learn_id=0, opponent id=1)
        """
        def __init__(self, *args):
            self.self_play = False
            self.algorithm = None

            # all agents
            self.agents = None

            # opponent to be changed
            self.opponents, self.weights = None, None

            self.model_path = None
            self.update_interval = 1
            self.save_interval = 0

            self.update_count = 0
            self.save_count = 0
            self.run_count = 0


            super(SelfPlayWrapper, self).__init__(*args)

        def self_play_update(self):
            self.save_count += 1
            self.run_count += 1

            if self.self_play:
                opponent = random.choices(self.opponents, weights=self.weights)[0]
                self.agents[1] = opponent
                self.game_board.set_agents(self.agents)

            # Change the opponent to current model
            if self.self_play and self.update_interval != 1:
                self.update_count += 1
                if self.update_count == self.update_interval:
                    self.agent.model.save(self.model_path)
                    model = self.algorithm.load(self.model_path)
                    for opponent in self.opponents:
                        if hasattr(opponent, "model"):
                            opponent.model = model

                    self.update_count = 0
                    return

            # Save the model periodically
            if self.save_interval and self.save_count >= self.save_interval:
                self.agent.model.save(self.model_path)
                self.save_count = 0

        def enable_self_play(self, algorithm, model, save_path, agents, opponents, weights=None, update_interval=1, ):
            agents[0].model = model
            for opponent in opponents:
                if hasattr(opponent, "model"):
                    opponent.model = model

            self.self_play = True
            self.algorithm = algorithm
            self.model_path = save_path
            self.agents = agents
            self.opponents, self.weights = opponents, weights
            self.update_interval = update_interval
            agents = [agents, opponents[0]] if not agents else agents

        def reset(self):
            self.self_play_update()
            obs = super(SelfPlayWrapper, self).reset()
            return obs

    return SelfPlayWrapper