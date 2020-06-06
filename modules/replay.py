import random
from collections import Iterable
import numpy as np
from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = {}
        self.position = 0

    def push(self, transition):
        self.memory[self.position] = transition
        if len(self.memory) < self.capacity:
            self.position += 1
        else:
            self.position = random.randrange(self.capacity)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, alpha=0.5):
        super().__init__(capacity)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def push(self, *args):
        idx = self.position
        super().push(*args)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self.memory) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta=0.4):
        assert beta > 0
        idxes = self._sample_proportional(batch_size)
        samples, weights = [], []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self.memory)) ** (-beta)

        for idx in idxes:
            samples.append(self.memory[idx])
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self.memory)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        return samples, weights, idxes

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.memory)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class EpisodicReplayBuffer:
    def __init__(self, capacity, num_workers, max_length):
        self.num_episodes = capacity // max_length
        self.max_length = max_length
        self.memory = {}
        self.position = 0
        self.trajectory = {i: [] for i in range(num_workers)}

    def push_transition(self, idx, transition):
        self.trajectory[idx].append(transition)
        if len(self.trajectory[idx]) == self.max_length:
            trajectory = np.asarray(self.trajectory[idx])
            self.memory[self.position] = trajectory
            self.trajectory[idx] = []

            if len(self.memory) < self.num_episodes:
                self.position += 1
            else:
                self.position = random.randrange(self.num_episodes)

    def push(self, transitions):
        if not isinstance(transitions, Iterable):
            transitions = [transitions]
        for idx, transition in enumerate(transitions):
            self.push_transition(idx, transition)

    def sample(self, batch_size):
        assert len(self.memory) >= batch_size
        idx = random.sample(self.memory.keys(), batch_size)
        trajs = np.asarray([self.memory[i] for i in idx])
        data = np.transpose(trajs, (1, 0, 2))
        return data

    def __len__(self):
        return len(self.memory)
