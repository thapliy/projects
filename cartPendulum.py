Code for OpenAI Gym ```
import gym
import numpy
import random

env = gym.make('CartPole-v0')
state_dimensions = 4
action_dimensions = 2
state_discretition = 2

reward_space = numpy.zeros([state_discretition, state_discretition, state_discretition, state_discretition, action_dimensions])

for i_episode in range(100):
    observation = env.reset()
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    for t in range(100):
        env.render()

        print("observation {}, reward {}".format(observation, reward))
        index_array = numpy.where(observation > 0, 1, 0)
        print("is positive? {}".format(index_array))
        explore = random.random() > 0.9
        print("explore? {}".format(explore))
        if explore:
            selected_action = env.action_space.sample()
        else:
            action_0_reward = reward_space[index_array[0], index_array[1], index_array[2], index_array[3], 0]
            action_1_reward = reward_space[index_array[0], index_array[1], index_array[2], index_array[3], 1]
            selected_action = 0
            if action_1_reward > action_0_reward:
                selected_action = 1
        print ("selected action: {}".format(selected_action))

        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

        new_index_array = numpy.where(observation > 0, 1, 0)
        expected_reward = max(reward_space[index_array[0], index_array[1], index_array[2], index_array[3]])

        total_reward = expected_reward + reward

        # implement exponentially degrading average of reward
        last_score = reward_space[index_array[0], index_array[1], index_array[2], index_array[3], selected_action]
        new_score = last_score * 0.9 + 0.1 * total_reward
        reward_space[index_array[0], index_array[1], index_array[2], index_array[3], selected_action] = new_score

print("reward_space: {}".format(reward_space))```