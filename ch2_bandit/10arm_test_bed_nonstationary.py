import numpy as np
import random 
import matplotlib.pyplot as plt
import operator 

class N_arm_test_bed:
    def __init__(self, non_greedy_prob):
        self.reward_dists = {i: [0, 1] for i in range(0, 10)}
        self.non_greedy_prob = non_greedy_prob
        self.actions = list(self.reward_dists.keys())
        self.action_estimations = []
        self.action_counts = []
        self.optimal_count = 0
        self.avg_rewards = []
        self.optimal_action_percents = []
        self.total_reward = 0

    def update_reward_dist(self):
        for i in range(0, 10):
            self.reward_dists[i][0] += (0.01 * np.random.randn())

    def get_reward(self, action):
        mu = self.reward_dists[action][0]
        variance = self.reward_dists[action][1]
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
            self.update_reward_dist()
            # print(self.reward_dists)
            action = self.get_action_with_prob(self.non_greedy_prob)

            reward = self.get_reward(action)
            self.total_reward += reward
            self.avg_rewards.append(self.total_reward/i)

            self.action_counts[action] += 1
            step_size = self.get_step_size(action)

            old_action_esimation = self.action_estimations[action]
            self.action_estimations[action] = old_action_esimation + step_size * (reward - old_action_esimation)
            
            max_value = self.reward_dists[max(self.reward_dists, key=(lambda key: self.reward_dists[key][0]))][0]
            current_action_value = self.reward_dists[action][0]
            if  max_value == current_action_value:
                self.optimal_count += 1

            self.optimal_action_percents.append((self.optimal_count/i) * 100)

        return self.avg_rewards, self.optimal_action_percents

count = 10000
test1 = N_arm_test_bed(0.1)
test1_avg_rewards, test1_optimal_percents = test1.maximize_reward(count)
test2 = N_arm_test_bed(0.01)
test2_avg_rewards, test2_optimal_percents = test2.maximize_reward(count)
test3 = N_arm_test_bed(0)
test3_avg_rewards, test3_optimal_percents = test3.maximize_reward(count)

#Show avg action value
plt.subplot(121)
plt.title('Avg action value')
plt.plot(range(0, count), test1_avg_rewards, label='0.1')
plt.plot(range(0, count), test2_avg_rewards, label='0.01')
plt.plot(range(0, count), test3_avg_rewards, label='0')
plt.legend()

plt.subplot(122)
#Show optimal action percent
plt.title('optimal action percent')
plt.plot(range(0, count), test1_optimal_percents, label='0.1')
plt.plot(range(0, count), test2_optimal_percents, label='0.01')
plt.plot(range(0, count), test3_optimal_percents, label='0')
plt.legend()
# plt.show()
plt.show(block=False)
input("Hit Enter To Close")
plt.close()

