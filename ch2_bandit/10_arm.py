from gym import spaces
from gym.utils import seeding
import numpy as np
import gym
import matplotlib.pyplot as plt


class BanditsEnv(gym.Env):
  metadata = {'render.modes': ['ansi']}

  def __init__(self, arms=1, mean=0, variance=1):
    self.arms = arms
    self.action_space = spaces.Discrete(self.arms)
    self.observation_space = spaces.Discrete(1)
    self.seed()
    self.q_mean_variance = {}
    for i in range(arms):
      self.q_mean_variance[i] = (self.np_random.normal(mean, variance), variance)

    items = list(self.q_mean_variance.items())
    self.sorted_q = sorted(items, key=lambda item: item[1][0], reverse=True)
    self.optimal_action = self.sorted_q[0][0]

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def random_walk(self, action):
    self.q[action] += self.np_random.normal(0, 0.01)
    return self.q[action]

  def step(self, action):
    assert self.action_space.contains(action)
    if action == -1:
      return 0, 0, True, {}
    # reward = self.random_walk(action)
    mean, variance = self.q_mean_variance[action]
    reward = self.np_random.normal(mean, variance)
    # observation, reward, done, info
    return 0, reward, False, {}

  def reset(self):
    return 0

  def render(self, mode="ansi"):
    if mode == "ansi":
      return self.sorted_q


class NonstationaryBandit:
  def __init__(self, env, epsilon=0.001):
    self.env = env
    self.n_arms = env.arms
    self.Q = np.zeros((self.n_arms))
    self.N = np.zeros((self.n_arms))
    self.e = epsilon
    self.average_rewards = []
    self.optimal_actions = []

  def execute(self, n_steps=1000, debug=False):
    self.n_steps = n_steps
    for step in range(n_steps):
      q = self.simpleBanditAlgo(step)
      weights = self.N / np.sum(self.N)
      if debug:
        print(step, q, self.N, weights)
      self.average_rewards.append(np.average(q, weights=weights))
      self.optimal_actions.append(self.N[self.env.optimal_action] / np.sum(self.N))
    return (self.average_rewards, self.optimal_actions)

  def simpleBanditAlgo(self, step):
      N = self.N
      Q = self.Q
      rand = np.random.rand(1)[0]
      if rand > self.e and step > 1:
        a = np.argmax(Q)
      else:
        a = np.random.randint(0, self.n_arms)
      obs, reward, done, info = self.env.step(a)
      N[a] += 1
      Q[a] = Q[a] + (1 / N[a]) * (reward - Q[a])
      return Q


plt.close()

n_steps = 1000
n_ensemble = 10
E = {}
rewards_by_step = np.zeros((n_ensemble, n_steps))
optimals_by_step = np.zeros((n_ensemble, n_steps))

es = [0, 0.1, 0.01]
for e in es:
  for i in range(n_ensemble):
    env = BanditsEnv(arms=10)
    bandit = NonstationaryBandit(env, epsilon=e)
    average_rewards, optimal_actions = bandit.execute(n_steps=n_steps)
    rewards_by_step[i] = average_rewards
    optimals_by_step[i] = optimal_actions
  E[e] = (np.average(rewards_by_step, axis=0), np.average(optimals_by_step, axis=0))

# Figure 2.2
plt.figure()
for e in es:
  plt.subplot(2, 1, 1)
  plt.plot(np.arange(n_steps), E[e][0], label="e=" + str(e))
plt.xlabel("Steps")
plt.ylabel("Average reward")
plt.legend()
for e in es:
  plt.subplot(2, 1, 2)
  plt.plot(np.arange(n_steps), E[e][1], label="e=" + str(e))
plt.xlabel("Steps")
plt.ylabel("% Optimal action")
plt.legend()
plt.show(False)
