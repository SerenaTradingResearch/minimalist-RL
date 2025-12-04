import itertools
from copy import deepcopy

import torch as tc
import torch.nn as nn
import torch.nn.functional as F

from .models import ActorCritic
from .utils import RLData


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
