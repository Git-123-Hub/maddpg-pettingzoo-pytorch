import numpy as np
import torch


class ReplayBuffer:
    """data structure where we store the agent's experience, and sample from them for the agent to learn"""

    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size

        self.state = np.zeros(int(capacity), dtype=object)
        self.action = np.zeros(int(capacity), dtype=object)
        self.reward = np.zeros(int(capacity), dtype=float)
        self.next_state = np.zeros(int(capacity), dtype=object)
        self.done = np.zeros(int(capacity), dtype=bool)  # type for done

        self._index = 0  # current position for adding new experience
        self._size = 0  # record the number of the all the experiences stored
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self):
        """clear all the experience that has been stored, reset the replay memory"""
        self.state.fill(0)
        self.action.fill(0)
        self.reward.fill(0)
        self.next_state.fill(0)
        self.done.fill(0)

        self._index = 0
        self._size = 0

    def add(self, state, action, reward, next_state, done):
        """ add an experience to the memory """
        self.state[self._index] = state
        self.action[self._index] = action
        self.reward[self._index] = reward
        self.next_state[self._index] = next_state
        self.done[self._index] = done

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, indices):
        # retrieve data, Note that the data stored is ndarray
        states = self.state[indices]
        actions = self.action[indices]
        rewards = self.reward[indices]
        next_states = self.next_state[indices]
        dones = self.done[indices]

        def transfer(data, first_dim=False):
            """
            transfer ndarray to torch.tensor,
            if `first_dim` is True, stack the ndarray so that the first dimension is `batch_size`,
            otherwise, the returned value is just a tensor with length of the original ndarray.
            """
            if first_dim:
                data = np.vstack(data)
            return torch.from_numpy(data).float().to(self.device)

        # NOTE that `states`, `actions`, `next_states` will be passed to network(nn.Module),
        # so the first dimension should be `batch_size`
        states = transfer(states, first_dim=True)  # torch.Size([batch_size, state_dim])
        actions = transfer(actions, first_dim=True)  # torch.Size([batch_size, action_dim])
        rewards = transfer(rewards)  # just a tensor with length: batch_size
        next_states = transfer(next_states, first_dim=True)  # Size([batch_size, state_dim])
        dones = transfer(dones)  # just a tensor with length: batch_size

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self._size
