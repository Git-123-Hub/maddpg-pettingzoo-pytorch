import numpy as np
from matplotlib import pyplot as plt
from pettingzoo.mpe import simple_adversary_v2, simple_crypto_v2, simple_push_v2, simple_reference_v2, \
    simple_speaker_listener_v3, simple_spread_v2, simple_tag_v2, simple_world_comm_v2


def get_env(name):
    """create env and return it with its full name"""
    if name == 'adversary':
        return simple_adversary_v2.parallel_env(N=2, max_cycles=25,
                                                continuous_actions=True), 'simple_adversary_v2'
    if name == 'crypto':
        return simple_crypto_v2.parallel_env(max_cycles=25, continuous_actions=True), 'simple_crypto_v2'
    if name == 'push':
        return simple_push_v2.parallel_env(max_cycles=25, continuous_actions=True), 'simple_push_v2'
    if name == 'reference':
        return simple_reference_v2.parallel_env(local_ratio=0.5, max_cycles=25,
                                                continuous_actions=True), 'simple_reference_v2'
    if name == 'speaker':
        return simple_speaker_listener_v3.parallel_env(max_cycles=25,
                                                       continuous_actions=True), 'simple_speaker_listener_v3'
    if name == 'spread':
        return simple_spread_v2.parallel_env(N=3, local_ratio=0.5, max_cycles=25,
                                             continuous_actions=True), 'simple_spread_v2'
    if name == 'tag':
        return simple_tag_v2.parallel_env(num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=25,
                                          continuous_actions=True), 'simple_tag_v2'
    if name == 'comm':
        return simple_world_comm_v2.parallel_env(num_good=2, num_adversaries=4, num_obstacles=1,
                                                 num_food=2, max_cycles=25, num_forests=2,
                                                 continuous_actions=True), 'simple_world_comm_v2'


class LinearDecayParameter:
    """parameter that decay linearly"""

    def __init__(self, x_0, y_0, x_1, y_1, *, min_value=None):
        """specify two points(x_0, y_0), (x_1, y_1) to calculate `y=kx+b`"""
        self.x_0 = x_0
        self.x_1 = x_1
        self.min_value = min_value  # used to clip value
        self.k = (y_1 - y_0) / (x_1 - x_0)
        self.b = y_1 - self.k * x_1

    def __call__(self, x):
        value = self.k * x + self.b
        if self.min_value is None:
            return value
        return max(value, self.min_value)

    def plot(self, x_0=None, x_1=None):
        if x_0 is None: x_0 = self.x_0
        if x_1 is None: x_1 = self.x_1

        if x_0 > x_1:
            x_0, x_1 = x_1, x_0  # ensure x_0 is the small one

        x = x_0
        x_values, y_values = [], []
        while x < x_1:
            y = self(x)
            x_values.append(x)
            y_values.append(y)
            x += 0.1
        fig, ax = plt.subplots()
        ax.plot(x_values, y_values)
        plt.show()


def get_running_reward(rewards: np.ndarray, window=100):
    """calculate the running reward, i.e. average of last `window` elements from rewards"""
    running_reward = np.zeros_like(rewards)
    for i in range(window - 1):
        running_reward[i] = np.mean(rewards[:i + 1])
    for i in range(window - 1, len(rewards)):
        running_reward[i] = np.mean(rewards[i - window + 1:i + 1])
    return running_reward


if __name__ == '__main__':
    p = LinearDecayParameter(0, 0.3, 500, 0, min_value=0)
    p.plot(0, 1000)
