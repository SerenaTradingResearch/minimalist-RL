import itertools
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from trading_models.utils import tensor, to_np

from .utils import RLData, mlp


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
        if with_logp:
            logp = dist.log_prob(act).sum(-1)
            # tanh squashing correction
            logp -= (2 * (np.log(2) - act - F.softplus(-2 * act))).sum(1)
            return act.tanh(), logp
        return act.tanh()

    @tc.no_grad()
    def tanh_act(s, obs, deter=False):
        return to_np(s(tensor(obs), deter, False))


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


class SAC:
    gamma = 0.99
    alpha = 0.2
    lr = 1e-3
    polyak = 0.995

    def __init__(s, ac: ActorCritic):
        s.ac = ac
        s.ac_tar = deepcopy(ac)
        for p in s.ac_tar.parameters():
            p.requires_grad = False
        s.pi_opt = tc.optim.Adam(ac.pi.parameters(), s.lr)
        s.q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
        s.q_opt = tc.optim.Adam(s.q_params, s.lr)

    def q_loss(s, d: RLData):
        with tc.no_grad():
            act2, logp2 = s.ac.pi(d.obs2)
            q_tar = tc.min(*s.ac_tar.q1_q2(d.obs2, act2))
            backup = d.rew + s.gamma * (1 - d.done) * (q_tar - s.alpha * logp2)
        return sum(F.mse_loss(q, backup) for q in s.ac.q1_q2(d.obs, d.act))

    def pi_loss(s, d: RLData):
        act, logp = s.ac.pi(d.obs)
        q = tc.min(*s.ac.q1_q2(d.obs, act))
        return tc.mean(s.alpha * logp - q)

    def update(s, d: RLData):
        s.q_opt.zero_grad()
        s.q_loss(d).backward()
        s.q_opt.step()
        for p in s.q_params:
            p.requires_grad = False

        s.pi_opt.zero_grad()
        s.pi_loss(d).backward()
        s.pi_opt.step()
        for p in s.q_params:
            p.requires_grad = True

        with tc.no_grad():
            for p, p_tar in zip(s.ac.parameters(), s.ac_tar.parameters()):
                p_tar.data.mul_(s.polyak)
                p_tar.data.add_((1 - s.polyak) * p.data)


def make_test_sac(env, ac: nn.Module, ac_tar: nn.Module):
    sac = SAC(ActorCritic(env, Act=nn.ReLU))
    sac.ac.load_state_dict(ac.state_dict())
    sac.ac_tar.load_state_dict(ac_tar.state_dict())
    return sac
