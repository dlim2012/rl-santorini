
def self_play_wrapper(cls):

    class SelfPlayWrapper(cls):
        """
        Only the second player changes
        """
        def __init__(self, *args):
            self.self_play = False
            self.algorithm = None

            self.agent = None
            self.opponent = None

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

            # Change the opponent to current model
            if self.self_play and self.update_interval != 1:
                self.update_count += 1
                if self.update_count == self.update_interval:
                    self.agent.model.save(self.model_path)
                    self.opponent.model = self.algorithm.load(self.model_path)

                    self.update_count = 0
                    return

            if self.save_interval and self.save_count >= self.save_interval:
                self.agent.model.save(self.model_path)
                self.save_count = 0

        def enable_self_play(self, algorithm, save_path, agent, opponent, model, update_interval, save_interval=100):
            self.model = agent.model = opponent.model = model

            self.self_play = True
            self.algorithm = algorithm
            self.model_path = save_path
            self.agent, self.opponent = agent, opponent
            self.update_interval = update_interval
            self.save_interval = save_interval if save_path else 0

        def reset(self):
            obs = super(SelfPlayWrapper, self).reset()
            self.self_play_update()
            return obs

    return SelfPlayWrapper