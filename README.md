
## Intro

- `Minimalist` & `Decoupled` Reinforcement Learning

![](https://raw.githubusercontent.com/NeuroAI-Research/minimalist-RL/main/test/SAC_HalfCheetah-v5.png)

## Usage

```bash
pip install minimalist-RL
```

```py
import gymnasium as gym
import torch.nn as nn

from minimalist_RL.SAC import SAC, ActorCritic
from minimalist_RL.utils import train_RL

env = gym.make("HalfCheetah-v5")
ac_net = ActorCritic(env, sizes=[256, 256], Act=nn.ReLU)
sac = SAC(ac_net)
train_RL(env, ac_net.pi.tanh_act, sac.update)
```
