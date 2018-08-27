from gym import spaces
from gym.utils import seeding
import numpy as np
import gym


class BanditsEnv(gym.Env):
  metadata = {'render.modes': ['ansi']}

  def __init__(self, arms=1, mean=0, variance=1):
    self.n = arms
    self.action_space = spaces.Discrete(self.n)
    self.observation_space = spaces.Discrete(1)
    self.seed()
    self.q_mean_variance = {}
    for i in range(arms):
      self.q_mean_variance[i] = (self.np_random.normal(mean, variance), variance)

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
    return 0, reward, False, {}

  def reset(self):
    return 0

  def render(self, mode="ansi"):
    if mode == "ansi":
      return self.q_mean_variance

n_steps = 1000
n_arms = 10
env = BanditsEnv(arms=n_arms)

Q = np.zeros((n_arms, n_steps))
N = np.zeros((n_arms, n_steps))
aciton = 0
e = 0.001

Q = np.array([[1,2], [3, 4]])

np.argmax(Q[:, 1])

np.random.randint(0, n_arms + 1)


for i in range(10):
  rand = np.random.rand(1)[0]
  if rand > e and i > 1:
    action = np.argmax(Q[:, i - 1])
  else:




  print(env.step(i))




