from typing import Dict, List

import numpy as np
import torch as tc
import torch.nn as nn


def transpose(matrix: List[List]):
    return list(zip(*matrix))


def set_seed(x=0):
    np.random.seed(x)
    tc.manual_seed(x)


def mlp(sizes, Act=nn.Tanh, out=[]):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), Act()]
    return nn.Sequential(*layers[:-1], *out)


class RLData:
    def __repr__(s):
        return f"ptr: {s.ptr}, size: {s.size}, cap: {s.cap}, keys: {s.keys}"

    def __init__(s, data: Dict = {}, cap=0, obs_dim=1, act_dim=1):
        def zeros(dim) -> tc.Tensor:
            return np.zeros((cap, dim), dtype=np.float32)

        if cap:
            s.obs = zeros(obs_dim)
            s.obs2 = zeros(obs_dim)
            s.act = zeros(act_dim)
            s.rew = zeros(1)
            s.done = zeros(1)
        s.__dict__.update(data)
        s.keys = list(s.__dict__.keys())
        s.ptr = 0
        s.size = len(data.get("obs", []))
        s.cap = s.size or cap

    def store(s, data: Dict):
        for k, v in data.items():
            getattr(s, k)[s.ptr] = v
        s.ptr = (s.ptr + 1) % s.cap
        s.size = min(s.size + 1, s.cap)

    def sample(s, n=32):
        idx = np.random.randint(0, s.size, n) if s.size else None
        return RLData({k: getattr(s, k)[idx] for k in s.keys})
