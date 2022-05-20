from pettingzoo.mpe import simple_adversary_v2, simple_spread_v2, simple_tag_v2


def get_env(name, ep_len=25):
    """create env and return it with its full name"""
    if name == 'adversary':
        return simple_adversary_v2.parallel_env(max_cycles=ep_len), 'simple_adversary_v2'
    if name == 'spread':
        return simple_spread_v2.parallel_env(max_cycles=ep_len), 'simple_spread_v2'
    if name == 'tag':
        return simple_tag_v2.parallel_env(max_cycles=ep_len), 'simple_tag_v2'
