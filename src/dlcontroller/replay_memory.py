"""
Originally implemented by: https://github.com/pranz24/pytorch-soft-actor-critic
Check LICENSE for details
"""
import random
import numpy as np


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, next_state, done):
        """Pushes the tuple into the memory and deletes oldest samples if the size of the memory
        has reached its maximum capacity.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        # print('next state shape in the memory: ', next_state.shape)
        self.buffer[self.position] = (state, action, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state, done = map(np.stack, zip(*batch))
        # print('reward shape in the replay memory: ', reward.shape)
        return state, action, next_state, done

    def __len__(self):
        return len(self.buffer)
