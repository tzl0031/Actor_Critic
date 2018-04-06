import random
import numpy as np
from collections import deque


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r, s2, done):
        # experience tuple
        experience = (s, a, r, s2, done)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

            s_batch = np.array([_[0] for _ in batch])
            a_batch = np.array([_[1] for _ in batch])
            r_batch = np.array([_[2] for _ in batch])
            s2_batch = np.array([_[3] for _ in batch])
            done_batch = np.array([_[4] for _ in batch])

            return s_batch, a_batch, r_batch, s2_batch, done_batch

    def clear(self):
        self.buffer = deque()
        self.count = 0
