import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from absl import app, flags
from functools import partial
import numpy as np
from numpy.random import choice
import jax
import tqdm
import gymnasium as gym
from gymnasium_robotics.envs.maze.point_maze import PointMazeEnv
import matplotlib.pyplot as plt
from env_helpers import GoalBonusRewardWrapper
import jaxrl_m.examples.mujoco.sac as learner

from jaxrl_m.wandb import setup_wandb, default_wandb_config, get_flag_dict
import wandb
from jaxrl_m.evaluation import supply_rng, evaluate, evaluate_with_trajectories_and_frames, flatten, EpisodeMonitor
from jaxrl_m.dataset import ReplayBuffer

from ml_collections import config_flags
import orbax.checkpoint as orbax
import time

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "customed_maze", "Environment name.")
flags.DEFINE_string("save_dir", None, "Logging dir.")
flags.DEFINE_integer("seed", np.random.choice(1000000), "Random seed.")
flags.DEFINE_integer("eval_episodes", 5, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("save_interval", 25000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(2.5e5), "Number of training steps.")
flags.DEFINE_integer("start_steps", int(1e4), "Number of initial exploration steps.")

wandb_config = default_wandb_config()
wandb_config.update(
    {
        "project": "d4rl_test",
        "group": "sac_test",
        "name": "sac_{env_name}",
    }
)

config_flags.DEFINE_config_dict("wandb", wandb_config, lock_config=False)
config_flags.DEFINE_config_dict("config", learner.get_default_config(), lock_config=False)

datetime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
root_path = os.getcwd()
ckpt_dir = "ckpt"
ckpt_path = os.path.join(root_path, ckpt_dir, datetime)

############################################
### Custom maze environment settings START
############################################
frame_dir = "custom_maze"
frame_path = os.path.join(root_path, frame_dir)
os.makedirs(frame_path, exist_ok=True)
# 定义迷宫
maze = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, "g", 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, "r", 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]
# 创建环境
env_ = PointMazeEnv(
    maze_map=maze,
    render_mode="rgb_array",
    continuing_task=False,
    # reward_type="dense",
)  # dense 奖励

obs, info = env_.reset()

# ✅ 设置摄像头（注意这里）
viewer = env_.point_env.mujoco_renderer._get_viewer("rgb_array")
viewer.cam.azimuth = 0
viewer.cam.elevation = -90
viewer.cam.distance = 15.0
viewer.cam.lookat[:] = [0, 0, 0]  # 中心点位置

# env_= GoalBonusRewardWrapper(env_, bonus=10.0, threshold=0.3)
env_ = gym.wrappers.TimeLimit(env_, max_episode_steps=1000)
############################################
### Custom maze environment settings END
############################################


def main(_):
    # Create wandb logger
    setup_wandb(FLAGS.config.to_dict(), **FLAGS.wandb)
    FLAGS.config.update({"target_entropy": -3})

    def metrics_fn(
        metrics: dict,
        bad_score=2e4,
        ideal_length=None,
        min_len=300,
        max_len=990,
        score_bar=0.5,
    ) -> float:
        # 改成你 metrics 实际的 key
        length = metrics.get("length", 0.0)
        score = metrics.get("score", 0.0)

        if length < min_len or length > max_len or score < score_bar:
            return bad_score

        if ideal_length is None:
            ideal_length = (max_len + min_len) / 2.0

        norm_length_error = (length - ideal_length) / (max_len - min_len)
        norm_score_error = score - 1.0
        return norm_length_error**2 + norm_score_error**2

    # options = orbax.CheckpointManagerOptions(
    #     max_to_keep=20,
    #     best_fn=metrics_fn,
    #     best_mode="min",
    #     enable_async_checkpointing=False,
    # )

    # checkpoint_manager = orbax.CheckpointManager(
    #     ckpt_path,
    #     orbax.PyTreeCheckpointer(),
    #     options=options,
    # )

    env = EpisodeMonitor(env_)
    eval_env = env

    example_transition = dict(
        observations=env.observation_space.sample()["observation"],
        actions=env.action_space.sample(),
        rewards=0.0,
        masks=1.0,
        next_observations=env.observation_space.sample()["observation"],
    )

    replay_buffer = ReplayBuffer.create(example_transition, size=int(1e6))

    agent = learner.create_learner(
        FLAGS.seed, example_transition["observations"][None], example_transition["actions"][None], max_steps=FLAGS.max_steps, **FLAGS.config
    )

    exploration_metrics = dict()
    obs = env.reset()
    exploration_rng = jax.random.PRNGKey(0)

    extra_eval = False
    extra_count = 0

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, dynamic_ncols=True):

        if i < FLAGS.start_steps:
            action = env.action_space.sample()
        else:
            exploration_rng, key = jax.random.split(exploration_rng)
            action = agent.sample_actions(obs, seed=key)

        next_obs, reward, done, info = env.step(action)
        mask = float(not done or "TimeLimit.truncated" in info)

        replay_buffer.add_transition(
            dict(
                observations=obs,
                actions=action,
                rewards=reward,
                masks=mask,
                next_observations=next_obs,
            )
        )
        obs = next_obs

        if done:
            exploration_metrics = {f"exploration/{k}": v for k, v in flatten(info).items()}
            obs = env.reset()

        if replay_buffer.size < FLAGS.start_steps:
            continue

        batch = replay_buffer.sample(FLAGS.batch_size)
        agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)
            wandb.log(exploration_metrics, step=i)
            exploration_metrics = dict()
            
        extra_count = extra_count + 1 if extra_eval else 0
        if i % FLAGS.eval_interval == 0 or (extra_eval and extra_count % 500 == 0):
            policy_fn = partial(supply_rng(agent.sample_actions), temperature=0.0)
            eval_info, _ = evaluate_with_trajectories_and_frames(
                policy_fn,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                train_step=i,
                dir_name=datetime,
            )
            eval_metrics = {f"evaluation/{k}": v for k, v in eval_info.items()}
            wandb.log(eval_metrics, step=i)

            metric_info = {
                "length": eval_info["episode.length"],
                "score": eval_info["episode.return"],
            }
            
            metric_value = metrics_fn(metric_info)
            extra_eval = metric_value < 2.0
            
            if extra_eval:
                print(f"Extra evaluation triggered: {i}, {metric_value}")


if __name__ == "__main__":
    app.run(main)
