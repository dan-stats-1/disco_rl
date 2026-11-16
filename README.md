# Continuous action space research

I adapted the DiscoRL approach to continuous action spaces utilising KDE-based density estimates and smoothed 1D Wasserstein-1 distances. It didn't learn an effective rule for continuous catch. See `continuous_disco_rl.md` for the method and results.

## Experiment 1: Confirm continuous catch is solvable

Firstly I want to make sure that continuous catch is set up correctly and can be solved by other less complex algorithms. I'll train a PPO agent and see how it fares:
[PPO results on continuous catch](results/continuous_catch_ppo/learning_curves.png)
[PPO episode](results/continuous_catch_ppo/episode_comparison.gif)

## Experiment 2: Checking wasserstein works

The next check is to see if smooth Wasserstein works, by freezing the meta network targets we then train the agent to see if it can learn parameters that minimise the overall loss but also each component of the loss function.
Results:
[Wasserstein performance on frozen meta targets](results/wasserstein_results.png)


## Experiment 3: KDE validation

The next thing is to check our next amendment to the existing methodology which uses KDE to estimate the densities of the policy samples. We check 3 different aspects

1. **Numerical correctness:** bounded-density integration, independent SciPy
   truncated-normal reference, support/permutation/JIT checks, first derivatives,
   directional second derivatives, and detached policy-gradient queries and
   advantages. Each check reports an explicit error, tolerance, and pass status.
2. **Statistical accuracy:** known central, narrow, and near-boundary
   tanh-Gaussian distributions; sample/bandwidth sweeps across seeds; density
   error, held-out log-density error, entropy self-sample bias, and exact versus
   estimated importance ratios. The ratio target is another known policy;
   query actions are drawn from the behaviour policy. These are measurements,
   not claims that KDE is an exact likelihood.
3. **Learning usefulness:** exact-likelihood score gradients versus KDE gradients
   on a one-step bounded task with reward `-(action - 0.6)**2`. Gauss-Hermite
   quadrature supplies an independent expected-return gradient and evaluates
   actual learning. Separate independent banks each supply one executed action;
   the actor-like KDE bank contains that action, while an independent-bank
   comparison isolates self-sample effects. Queries and advantages stay detached.

[Numerical correctness results](results\kde_validation\numerical\validation.png)
[Statistical accuracy results](results\kde_validation\accuracy\validation.png)
[Learning usefulness results](results\kde_validation\learning\validation.png)

## Experiment 4: Direct PPO policy imitation

The next step was to check if our agent architecture is even capable of learning a PPO like policy. 
To do this we will create an agent and use our frozen PPO policy and then use smooth wasserstein between the agent samples and samples from the PPO policy.

$$L(\theta) = \mathbb{E}\left[ W_{smooth}(\mathbf{\pi}_{\theta}(s), \mathbf{\pi}_{ppo}(s)) \right]$$

[Imitation learning results](results\continuous_catch_ppo_imitation\learning_curves.png)

## Experiment 5(a): PPO-guided meta-learning

The next check is to see if the agent can learn PPO-like policies from meta network targets but guided by the successful PPO policy. 

A newly initialised Disco meta-network supplies the inner policy targets.
The existing differentiable agent learner/optimiser performs several inner
updates. The outer objective is smooth Wasserstein distance between the updated
agent and frozen PPO on saved validation observations, replacing A2C entirely.
There are no entropy/target regularizers or `y/z/p/q` loss contributions.

So the agent loss function becomes 
$$L(\theta) = \mathbb{E}_{s,a \sim \mathbf{\pi}_{\theta}} \left[W_{smooth}(\mathbf{\pi}, \mathbf{\pi}_{\theta}(s)) \right] $$

The meta loss function becomes
$$L(\eta) = \mathbb{E}\left[ W_{smooth}(\mathbf{\pi}_{\theta}(s), \mathbf{\pi}_{ppo}(s)) \right]$$

So this encourages the meta network to produce targets that result in agents learning a PPO-like policy.

At the end, three fresh students learn from the **frozen discovered rule**, without PPO supplying any training targets.
Their before/after catch rates test whether the rule works beyond the students used to
train it. PPO is queried only to measure policy distance in these evaluations.

[PPO guided meta-learning results](results\continuous_catch_ppo_meta\learning_curves.png)

This didn't train successfully so further investigation is required.

## Experiment 5(b): PPO-guided meta-learning with full meta network targets

Given the previous experiment wasn't successful, we try the same experiment but with `y, z, q, p` targets included in to check if these targets are required for the agent to learn a PPO-like policy. 

The outer objective remains **only** Wasserstein distance to frozen PPO, with no
A2C or extra regularizers. Inner policy, `y`, and `z` targets come from Disco.
The auxiliary-policy `p` target is the detached next-state agent policy; `q`
uses the original detached TD/value target. These are the original inner
objectives, not extra targets supplied by PPO. Reward therefore participates
in the restored inner value learning, but not directly in the outer loss.

The same full inner loss is used when testing the frozen learned rule on fresh
agents.

[PPO guided meta-learning with full meta network targets results](results\continuous_catch_ppo_meta_full\learning_curves.png)

# DiscoRL: Discovering State-of-the-art Reinforcement Learning Algorithms

This repository contains accompanying code for the *"Discovering
 State-of-the-art Reinforcement Learning Algorithms"* Nature publication.

It provides a minimal JAX harness for the DiscoRL setup together with the
 original meta-learned weights for the *Disco103* discovered update rule.

The harness supports both:

-   **Meta-evaluation**: training an agent using the *Disco103* discovered RL
    update rule, using the `colabs/eval.ipynb` notebook [![Open In](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/disco_rl/blob/master/colabs/eval.ipynb) and

-   **Meta-training**: meta-learning a RL update rule from scratch or from a
    pre-existing checkpoint, using the `colabs/meta_train.ipynb` notebook [![Open In](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/disco_rl/blob/master/colabs/meta_train.ipynb)

Note that it will not be actively maintained moving forward.

## Installation

Set up a Python virtual environment and install the package:

```bash
python3 -m venv disco_rl_venv
source disco_rl_venv/bin/activate
pip install git+https://github.com/google-deepmind/disco_rl.git
```

The package can also be installed from colab:

```bash
!pip install git+https://github.com/google-deepmind/disco_rl.git
```

## Usage

The code is structured as follows:

* `environments/` contains the general interface for the environments that can
  be used with the provided harness, and two implementations of `Catch`:
  a CPU-based one and jittable;

* `networks/` includes a simple MLP network and LSTM-based components of the
  DiscoRL models, all implemented in Haiku;

* `update_rules/` has implementations of the discovered rules, actor-critic, and
  policy gradient;

* `value_fns/` contains value-function related utilities;

* `types.py`, `utils.py`, `optimizers.py` implement a basic functionality for
  the harness;

* `agent.py` is a generic implementation of an RL agent which uses the update
  rule's API for training, hence it is compatible with all the rules from
  `update_rules/`.

Detailed examples of usage can be found in the colabs above.

## Citation

Please cite the original Nature paper:

```
@Article{DiscoRL2025,
  author  = {Oh, Junhyuk and Farquhar, Greg and Kemaev, Iurii and Calian, Dan A. and Hessel, Matteo and Zintgraf, Luisa and Singh, Satinder and van Hasselt, Hado and Silver, David},
  journal = {Nature},
  title   = {Discovering State-of-the-art Reinforcement Learning Algorithms},
  year    = {2025},
  doi     = {10.1038/s41586-025-09761-x}
}
```

## License and disclaimer

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
