from typing import Callable

import gymnasium as gym
import numpy as np
import torch as tc
import torch.nn as nn
from spinup.sac import (
    MLPActorCritic,
    MLPQFunction,
    ReplayBuffer,
    SquashedGaussianMLPActor,
)
from trading_models.utils import plot_general, shape

from minimalist_RL.SAC import SAC, ActorCritic, NormalActor, QFunc
from minimalist_RL.utils import ActMap, RLData, mlp, set_seed


def learn_RL_env(
    env: gym.Env,
    get_tanh_act: Callable,
    update_model: Callable,
    steps=1e5,
    rand_steps=1e3,
    batch_size=100,
):
    sp = env.action_space
    obs = env.reset()[0]
    data, records, score = RLData(), RLData(), 0
    for t in range(int(steps)):
        if t == rand_steps:
            print("random exploration ended")
        if t < rand_steps:
            env_act = sp.sample()
            act = ActMap.to_tanh(env_act, sp.low, sp.high)
        else:
            act = get_tanh_act(obs)
            env_act = ActMap.from_tanh(act, sp.low, sp.high)
        obs2, rew, term, trunc, _ = env.step(env_act)
        data.push(dict(obs=obs, obs2=obs2, act=act, rew=rew, done=term))
        score += rew
        obs = obs2
        if term or trunc:
            obs = env.reset()[0]
            records.push(dict(score=score))
            print(f"t: {t}, score: {score}")
            score = 0
        if data.size > batch_size and t % batch_size == 0:
            for _ in range(batch_size):
                update_model(data.sample(batch_size))
            if records.size:
                plot_general(records.dict(), env.spec.id)


def test1():
    n1 = SquashedGaussianMLPActor(3, 4, [5], nn.Tanh, 1)
    n2 = NormalActor([3, 5, 4], nn.Tanh)
    n2.load_state_dict(n1.state_dict())
    obs = tc.rand(10, 3)
    for n in [n1, n2]:
        set_seed()
        act, logp = n(obs)
        print(act.sum(), logp.sum())


def test2():
    n1 = MLPQFunction(3, 4, [5], nn.Tanh)
    n2 = QFunc(mlp([3 + 4, 5, 1], nn.Tanh))
    n2.load_state_dict(n1.state_dict())
    obs, act = tc.rand(10, 3), tc.rand(10, 4)
    print(n1(obs, act).sum())
    print(n2(obs, act).sum())


def test3():
    env = gym.make("InvertedPendulum-v5")
    n1 = MLPActorCritic(env.observation_space, env.action_space)
    n2 = ActorCritic(env, Act=nn.ReLU)
    n2.load_state_dict(n1.state_dict())
    print(n1)
    print(n2)


def test4():
    obs_dim, act_dim, cap = (3, 4), (5, 6), int(1e6)
    b1 = ReplayBuffer(obs_dim, act_dim, cap)
    b2 = RLData({}, cap)
    rand = np.random.rand
    for _ in range(1000):
        data = {
            "obs": rand(*obs_dim),
            "act": rand(*act_dim),
            "rew": rand(),
            "obs2": rand(*obs_dim),
            "done": rand(),
        }
        b1.store(*data.values())
        b2.push(data)
    np.random.seed(0)
    d1 = b1.sample_batch(100)
    np.random.seed(0)
    d2 = b2.sample(100)
    print({k: v.sum() for k, v in d1.items()})
    print({k: v.sum() for k, v in d2._data.items()})
    print(shape(d1))
    print(d2)


if __name__ == "__main__":
    # env = gym.make("HalfCheetah-v5")
    env = gym.make("InvertedPendulum-v5")
    ac_net = ActorCritic(env)
    sac = SAC(ac_net)
    learn_RL_env(
        env,
        ac_net.pi.tanh_act,
        sac.update,
    )
