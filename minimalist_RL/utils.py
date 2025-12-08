from typing import Callable, Dict

import gymnasium as gym
import numpy as np
import torch as tc
import torch.nn as nn
from trading_models.utils import plot_general, shape, tensor


def set_seed(x=0):
    np.random.seed(x)
    tc.manual_seed(x)


def mlp(sizes, Act=nn.Tanh, out=[]):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), Act()]
    return nn.Sequential(*layers[:-1], *out)


class ActMap:
    @staticmethod
    def from_tanh(tanh, low, high):
        return (tanh + 1) / 2 * (high - low) + low

    @staticmethod
    def to_tanh(x, low, high):
        return (x - low) / (high - low) * 2 - 1


class DataBuffer:
    def __repr__(s):
        return f"ptr: {s.ptr}, size: {s.size}/{s.cap} {shape(s._data)}"

    def __init__(s, data: Dict[str, tc.Tensor] = None, cap=1e6):
        s._data, s.cap = data or {}, int(cap)
        s.ptr, s.size = 0, 0
        if len(s._data):
            v0 = list(data.values())[0]
            s.cap = s.size = len(v0)

    def __getattr__(s, k):
        return s._data[k][: s.size]

    def dict(s):
        return {k: v[: s.size] for k, v in s._data.items()}

    def push(s, row: Dict):
        for k, v in row.items():
            if k not in s._data:
                shape = () if np.isscalar(v) else v.shape
                s._data[k] = np.full((s.cap, *shape), np.nan, dtype=np.float32)
            s._data[k][s.ptr] = v
        s.ptr, s.size = (s.ptr + 1) % s.cap, min(s.size + 1, s.cap)

    def sample(s, n=100):
        idx = np.random.randint(0, s.size, (n,)) if s.size else None
        return DataBuffer(tensor({k: v[idx] for k, v in s._data.items()}))


class RLData(DataBuffer):
    obs: tc.Tensor
    obs2: tc.Tensor
    act: tc.Tensor
    rew: tc.Tensor
    done: tc.Tensor


def train_RL(
    env: gym.Env,
    get_tanh_act: Callable,
    update_model: Callable,
    steps=1e6,
    rand_steps=1e3,
    batch_size=100,
):
    sp = env.action_space
    obs = env.reset()[0]
    data, records, score = RLData(), RLData(), 0
    for t in range(int(steps)):
        if t < rand_steps:
            env_act = sp.sample()
            act = ActMap.to_tanh(env_act, sp.low, sp.high)
        else:
            act = get_tanh_act(obs)
            env_act = ActMap.from_tanh(act, sp.low, sp.high)
        obs2, rew, term, trunc, _ = env.step(env_act)
        data.push(dict(obs=obs, obs2=obs2, act=act, rew=rew, done=term))
        obs, score = obs2, score + rew
        if term or trunc:
            records.push(dict(score=score, step=t, is_rand_act=t < rand_steps))
            id = env.spec.id if env.spec else env.__class__.__name__
            plot_general(records.dict(), id)
            print(f"step: {t}, score: {score}")
            obs, score = env.reset()[0], 0
        if t and t % batch_size == 0:
            for _ in range(batch_size):
                update_model(data.sample(batch_size))
