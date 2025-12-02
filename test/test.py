import gymnasium as gym
import numpy as np
import torch as tc
import torch.nn as nn
from torch.distributions import Categorical
from trading_models.utils import plot_general, tensor, to_np

from minimalist_RL.PPO import gae_adv_ret, ppo_loss
from minimalist_RL.utils import transpose


class ActorCritic(nn.Module):
    def __init__(s, obs_dim, act_dim):
        super().__init__()
        s.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        s.pi = nn.Linear(64, act_dim)
        s.v = nn.Linear(64, 1)

    def act(s, obs, act=None):
        h = s.shared(tensor(obs))
        pi, v = s.pi(h), s.v(h).squeeze()
        dist = Categorical(logits=pi)
        if act is None:
            act = dist.sample()
        return act, dist.log_prob(act.long()), dist.entropy(), v


@tc.no_grad()
def rollout(env: gym.Env, net: ActorCritic, steps=5000):
    obs = env.reset()[0]
    records = []
    score, scores = 0, []
    for _ in range(steps):
        act, logp, ent, val = map(to_np, net.act(obs))
        next_obs, rew, term, trunc, _ = env.step(act)
        next_val = 0.0 if term else to_np(net.act(next_obs)[-1])
        records.append([obs, act, logp, rew, term, val, next_val])
        end = term or trunc
        obs = env.reset()[0] if end else next_obs
        score += rew
        if end:
            print(f"score: {score}")
            scores.append(score)
            score = 0
    return [np.array(x) for x in transpose(records)], scores


def run_RL(env: gym.Env, epochs=10):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    net = ActorCritic(obs_dim, act_dim)
    opt = tc.optim.Adam(net.parameters(), lr=1e-3)
    all_scores, losses = [], []

    while True:
        records, scores = rollout(env, net)
        all_scores += scores
        if np.min(scores) == np.max(scores):
            break
        rew, term, val, next_val = records[-4:]
        adv, ret = tensor(gae_adv_ret(rew, term, val, next_val))
        obs, act, logp_old, rew, term, val, next_val = tensor(records)

        for _ in range(epochs):
            _, logp, ent, val = net.act(obs, act)
            loss = ppo_loss(logp, logp_old, adv, ret, val, ent)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss)
    plot_general({"loss": losses, "score": all_scores}, "PPO")


tc.manual_seed(0)
np.random.seed(0)
run_RL(gym.make("CartPole-v1"))
