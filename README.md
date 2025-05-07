# README

Official implementation of **Advantage-based Diffusion Actor-Critic (ADAC)** using `JAX` and `Flax`.

---

## Installation

### D4RL

```sh
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```

### JAX & Core Libraries

```sh
pip install "jax[cuda]" flax optax orbax distrax
```

### Utilities

```sh
pip install ml_collection wandb
```

### Customized `jaxrl_m`

```sh
cd jaxrl_m
pip install -e .
cd -
```

---

## Running the Code

The entry point is `adac_run.py`.
To train on a specific environment, run:

```sh
python adac_run.py --env_name <env_name>
```

### Tested Environments

We have evaluated ADAC on the following D4RL datasets:

```text
# Locomotion (MuJoCo)
walker2d-random-v2
walker2d-medium-v2
walker2d-medium-replay-v2
walker2d-medium-expert-v2
hopper-random-v2
hopper-medium-v2
hopper-medium-replay-v2
hopper-medium-expert-v2
halfcheetah-random-v2
halfcheetah-medium-v2
halfcheetah-medium-replay-v2
halfcheetah-medium-expert-v2

# AntMaze
antmaze-umaze-v0
antmaze-umaze-diverse-v0
antmaze-medium-play-v0
antmaze-medium-diverse-v0
antmaze-large-play-v0
antmaze-large-diverse-v0

# Adroit
pen-human-v1
pen-cloned-v1

# Kitchen
kitchen-complete-v0
kitchen-partial-v0
kitchen-mixed-v0
```

---

## PointMaze Demonstration

We also provide a toy offline environment (`PointMaze`) and SAC-based data collection scripts.

### Setup

Navigate to the `pointmaze` directory:

```sh
cd pointmaze
```

> ⚠️ **Do not install D4RL** for this demo.

Install dependencies:

```sh
pip install "jax[cuda]" flax optax orbax distrax
pip install ml_collection wandb
```

Install the customized `jaxrl_m` (modified for `gymnasium` and trajectory recording):

```sh
cd jaxrl_m
pip install -e .
cd -
```

### Run the Script

To run SAC on the custom maze environment:

```sh
python sac_custom_maze.py
```

## Acknowledgements
This repository builds upon the excellent [jaxrl_m](https://github.com/dibyaghosh/jaxrl_m) codebase. We thank the authors for their clean and modular implementation, which served as a foundation for our work.