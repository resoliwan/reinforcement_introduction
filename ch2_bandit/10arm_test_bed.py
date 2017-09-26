import numpy as np
import random 
import matplotlib.pyplot as plt
import operator 

reward_dists = {
    #action: (mean, variance)
    0: (1, 1),
    1: (-1, 1),
    2: (1.5, 1),
    3: (.2, 1),
    4: (1.2, 1),
    5: (-1.9, 1),
    6: (-.2, 1),
    7: (-1.1, 1),
    8: (.9, 1),
    9: (.9, 1)
}

class N_arm_test_bed:
    def __init__(self, reward_dists, non_greedy_prob):
        self.non_greedy_prob = non_greedy_prob
        self.actions = list(reward_dists.keys())
        self.action_estimations = []
        self.action_counts = []
        self.reward_dists = reward_dists
        self.max_action = max(reward_dists, key=(lambda key: reward_dists[key][0]))
        self.avg_rewards = []
        self.optimal_action_percent = []
        self.total_reward = 0

    def get_reward(self, action):
        mu = reward_dists[action][0]
        variance = reward_dists[action][1]
        return np.sqrt(variance) * np.random.randn() + mu
    
    def initalize(self):
        for _ in self.actions:
            self.action_estimations.append(0)
            self.action_counts.append(0)

    def get_action_with_prob(self, prob):
        if np.random.rand() > prob:
            max_estimation = max(self.action_estimations)
            # print('self.action_estimations', self.action_estimations)
            # print('max_estimation', max_estimation)
            # print('self.action_estimations.index(max_estimation)', self.action_estimations.index(max_estimation))
            return self.action_estimations.index(max_estimation)
        else:
            return random.choice(self.actions)

    def get_step_size(self, action):
        return 1/self.action_counts[action]

    def maximize_reward(self, count):
        self.initalize()
        for i in range(1, count+1):
            action = self.get_action_with_prob(self.non_greedy_prob)

            reward = self.get_reward(action)
            self.total_reward += reward
            self.avg_rewards.append(self.total_reward/i)

            self.action_counts[action] += 1
            step_size = self.get_step_size(action)

            old_action_esimation = self.action_estimations[action]
            self.action_estimations[action] = old_action_esimation + step_size * (reward - old_action_esimation)





        return self.avg_rewards

count = 10000
test1 = N_arm_test_bed(reward_dists, 0.1)
test1_average_rewards = test1.maximize_reward(count)
test2 = N_arm_test_bed(reward_dists, 0.01)
test2_average_rewards = test2.maximize_reward(count)
test3 = N_arm_test_bed(reward_dists, 0)
test3_average_rewards = test3.maximize_reward(count)

plt.plot(range(0, count), test1_average_rewards, label='0.1')
plt.plot(range(0, count), test2_average_rewards, label='0.01')
plt.plot(range(0, count), test3_average_rewards, label='0')
plt.legend()
plt.show()


