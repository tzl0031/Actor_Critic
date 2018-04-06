import gym
from gym import spaces
import numpy as np
from gym.utils import seeding


class MIMOEnv(gym.Env):
    metadata = {}

    def __init__(self):
        self.p_min = 0
        self.p_max = 1
        self.q_min = -1
        self.q_max = 1
        self.capacity = 20
        self.alpha = 1
        self.pd = 1

        self.H_p = np.random.uniform(self.p_min, self.p_max, size=(9, 1))
        self.H_q = np.random.uniform(self.q_min, self.q_max, size=(9, 1))
        self.HH_ = np.square(self.H_p) + np.square(self.H_q)
        # 2D action space
        self.action_space = spaces.Box(low=np.array((-1, -1)), high=np.array((1, 1)), dtype=float)
        # 3D observation_space
        self.observation_space = spaces.Box(np.array((self.p_min, self.q_min)),
                                            np.array((self.p_max, self.q_max)),
                                            dtype=float)
        self.state = None
        self.seed()

        high = np.array([1, 1])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # return
    def step(self, action):
        """

        :param action: 2-dim action
        :return: Next state, reward, done or not, {}
        """
        p, q = self.state
        new_p = np.clip(p + action[0], self.p_min, self.p_max)
        new_q = np.clip(q + action[1], self.q_min, self.q_max)
        # print("new_state")
        # print(new_p, new_q)

        hh_ = new_p**2 + new_q**2
        self.last_action = action
        snr = self.alpha * self.pd * hh_/(self.alpha * self.pd * np.sum(self.HH_) + 1)
        if abs(self.capacity - snr) < 0.1:
            reward = 1
        else:
            reward = -abs(self.capacity - snr)

        self.state = np.array([new_p, new_q])

        return self.state, reward, False, {}


    def reset(self):
        """
        Generate state from Uniform
        :return: 3-dim state
        """

        p = self.np_random.uniform(self.p_min, self.p_max)
        q = self.np_random.uniform(self.q_min, self.q_max)
        # print("reset")
        # print(p, q)
        # hh_ = p**2 + q**2
        # SNR = self.alpha * self.pd * hh_/(self.alpha * self.pd * np.sum(self.HH_) + 1)
        self.state = np.array([p, q])
        self.last_action = None

        return self.state


