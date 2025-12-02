from typing import List

import numba
import numpy as np
import torch as tc


@numba.njit
def gae_adv_ret(rew, term, val, next_val, gamma=0.99, lam=0.95):
    T = len(rew)
    adv = np.zeros(T)
    last = 0.0
    for t in range(T - 1, -1, -1):
        delta = rew[t] + gamma * next_val[t] * (1 - term[t]) - val[t]
        adv[t] = last = delta + gamma * lam * (1 - term[t]) * last
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ret = adv + val
    return adv, ret


def ppo_loss(
    logp: tc.Tensor,
    logp_old: tc.Tensor,
    adv: tc.Tensor,
    ret: tc.Tensor,
    val: tc.Tensor,
    ent: tc.Tensor,
    clip=0.2,
    weights: List[float] = [1.0, 0.5, 0.01],
):
    ratio = (logp - logp_old).exp()
    sur1 = ratio * adv
    sur2 = tc.clamp(ratio, 1 - clip, 1 + clip) * adv
    w1, w2, w3 = weights
    actor_loss = w1 * -tc.min(sur1, sur2).mean()
    critic_loss = w2 * (ret - val).pow(2).mean()
    ent_loss = w3 * -ent.mean()
    return actor_loss + critic_loss + ent_loss
