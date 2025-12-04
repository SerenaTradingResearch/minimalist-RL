from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .utils import mlp


class NormalActor(nn.Module):
    def __init__(s, sizes, Act=nn.Tanh):
        super().__init__()
        s.net = mlp(sizes[:-1], Act, [Act()])
        s.mu_layer = nn.Linear(sizes[-2], sizes[-1])
        s.log_std_layer = nn.Linear(sizes[-2], sizes[-1])

    def forward(s, obs, deter=False, with_logp=True):
        x = s.net(obs)
        mu = s.mu_layer(x)
        log_std = tc.clamp(s.log_std_layer(x), -20, 2)
        dist = Normal(mu, log_std.exp())
        act = mu if deter else dist.rsample()
        logp: tc.Tensor = tc.nan
        if with_logp:
            logp = dist.log_prob(act).sum(-1)
            # magic: Tanh squashing correction
            logp -= (2 * (np.log(2) - act - F.softplus(-2 * act))).sum(1)
        return act.tanh(), logp


class QFunc(nn.Module):
    def __init__(s, net: nn.Module):
        super().__init__()
        s.q = net

    def forward(s, obs, act):
        return s.q(tc.cat([obs, act], -1)).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(s, env: gym.Env, sizes=[256, 256], Act=nn.Tanh):
        super().__init__()
        obs_dim = env.observation_space.shape[0]
        x = env.action_space
        act_dim, s.low, s.high = x.shape[0], x.low, x.high
        s.pi = NormalActor([obs_dim, *sizes, act_dim], Act)
        s.q1 = QFunc(mlp([obs_dim + act_dim, *sizes, 1], Act))
        s.q2 = deepcopy(s.q1)

    def q1_q2(s, obs, act):
        return s.q1(obs, act), s.q2(obs, act)
