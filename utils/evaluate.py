from utils.tools import predict_with_mask

def evaluate_policy(model, env, discount_factor=1.0, n_eval_episodes=1000, print_result=False, agents=None, learn_id=None, count_invalid_actions=0):
    """

    :param model: model
    :param env: environment
    :param discount_factor: discount factor
    :param n_eval_episodes: number of episodes to evaluate
    :param print_result: print results if True
    :param agents, learn_id: change the agents
    :param count_invalid_actions: count invalid up to the given number
    :return:
    """
    count_invalid_actions *= n_eval_episodes

    prev_agents, prev_learn_id, self_play = [agent for agent in env.game_board.agents], env.game_board.learn_id, env.self_play
    env.game_board.set_agents(agents, learn_id)
    env.self_play = False

    counts = {'win': 0, 'tie': 0, 'lose': 0}
    total_rewards, invalid_action_count = 0, 0
    for i in range(n_eval_episodes):
        discounted_rewards = 0
        obs, done = env.reset(), False
        while not done:
            if invalid_action_count < count_invalid_actions:
                action, _state = model.predict(obs)
                while env.game_board.action_mask[action] == 0:
                    invalid_action_count += 1
                    if invalid_action_count == count_invalid_actions:
                        action = predict_with_mask(model, obs, env.game_board)
                        break
                    action, _state = model.predict(obs)
            else:
                action = predict_with_mask(model, obs, env.game_board)
            obs, rewards, done, info = env.step(action)
            discounted_rewards = discounted_rewards * discount_factor + rewards
        counts[info['result']] += 1
        total_rewards += discounted_rewards
    avg_rewards = total_rewards / n_eval_episodes

    # print results
    if print_result:
        print(
            "win %d, tie %d, lose %d, rewards %.2f" % \
            (counts['win'], counts['tie'], counts['lose'], avg_rewards)
        )
        if count_invalid_actions:
            if invalid_action_count == count_invalid_actions:
                print('Invalid actions >= %.2f' % (invalid_action_count / n_eval_episodes))
            else:
                print('Invalid actions = %.2f' % (invalid_action_count / n_eval_episodes))

    env.game_board.set_agents(prev_agents, prev_learn_id)
    env.self_play = self_play
    counts['invalid_action_count'] = invalid_action_count / n_eval_episodes
    return avg_rewards, counts


