from typing import Dict, List, Tuple
import jax
import gymnasium as gym
import numpy as np
from collections import defaultdict
import time
import os
import cv2
import pickle
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

work_dir = os.getcwd()


class MazeVisualizer:
    def __init__(self, env, root_dir, frame_dir, traj_dir):
        self.env = env
        self.root_dir = root_dir
        self.frame_dir = os.path.join(root_dir, frame_dir)
        self.traj_dir = os.path.join(root_dir, traj_dir)
        os.makedirs(self.frame_dir, exist_ok=True)
        os.makedirs(self.traj_dir, exist_ok=True)

    def save_frame(self, step):
        """保存当前帧图像为 frame_xxx.png，env.render() 返回 RGB 图"""
        img = self.env.render()
        plt.imsave(os.path.join(self.frame_dir, f"frame_{step:03d}.png"), img)

    def save_traj_data(self, data):
        """保存轨迹数据为 pickle 文件"""
        datetime_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        with open(os.path.join(self.traj_dir, f"trajectory_data_{datetime_str}.pkl"), "wb") as f:
            pickle.dump(data, f)

    def draw_traj(self):
        """从帧图像中提取绿色 agent 轨迹并绘制"""

        frames = sorted([f for f in os.listdir(self.frame_dir) if f.endswith(".png")])
        if not frames:
            print("No frame images found in:", self.frame_dir)
            return

        positions = []

        for frame_file in frames:
            img_path = os.path.join(self.frame_dir, frame_file)
            img = cv2.imread(img_path)  # BGR
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # 绿色阈值（保持你原来的）
            lower_green = (40, 40, 40)
            upper_green = (80, 255, 255)
            mask = cv2.inRange(hsv, lower_green, upper_green)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    positions.append((cx, cy))

        if not positions:
            print("No green agent found in any frame.")
            return

        # 用最后一帧画轨迹
        last_frame = cv2.imread(os.path.join(self.frame_dir, frames[-1]))
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))

        if len(positions) > 1:
            xs, ys = zip(*positions)
            ax.plot(xs, ys, "r.-", linewidth=2, markersize=4)

        ax.axis("off")
        fig.tight_layout()
        datetime_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        fig.savefig(os.path.join(self.traj_dir, f"final_trajectory_{datetime_str}.png"))
        plt.close(fig)


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """
    Wrapper that supplies a jax random key to a function (using keyword `seed`).
    Useful for stochastic policies that require randomness.

    Similar to functools.partial(f, seed=seed), but makes sure to use a different
    key for each new call (to avoid stale rng keys).

    """

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key="", sep="."):
    """
    Helper function that flattens a dictionary of dictionaries into a single dictionary.
    E.g: flatten({'a': {'b': 1}}) -> {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(policy_fn, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    """
    Evaluates a policy in an environment by running it for some number of episodes,
    and returns average statistics for metrics in the environment's info dict.

    If you wish to log environment returns, you can use the EpisodeMonitor wrapper (see below).

    Arguments:
        policy_fn: A function that takes an observation and returns an action.
            (if your policy needs JAX RNG keys, use supply_rng to supply a random key)
        env: The environment to evaluate in.
        num_episodes: The number of episodes to run for.
    Returns:
        A dictionary of average statistics for metrics in the environment's info dict.

    """
    stats = defaultdict(list)
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action = policy_fn(observation)
            observation, _, done, info = env.step(action)
            add_to(stats, flatten(info))
        add_to(stats, flatten(info, parent_key="final"))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats


def evaluate_with_trajectories(
    policy_fn,
    env: gym.Env,
    num_episodes: int,
) -> Tuple[Dict[str, float], List[Dict[str, List]]]:
    """
    Same as evaluate, but also returns the trajectories of observations, actions, rewards, etc.

    Arguments:
        See evaluate.
    Returns:
        stats: See evaluate.
        trajectories: A list of dictionaries (each dictionary corresponds to an episode),
            where trajectories[i] = {
                'observation': list_of_observations,
                'action': list_of_actions,
                'next_observation': list_of_next_observations,
                'reward': list of rewards,
                'done': list of done flags,
                'info': list of info dicts,
            }
    """

    trajectories = []
    stats = defaultdict(list)

    for _ in range(num_episodes):
        trajectory = defaultdict(list)
        observation, done = env.reset(), False
        while not done:
            action = policy_fn(observation)
            next_observation, r, done, info = env.step(action)
            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=r,
                done=done,
                info=info,
            )
            add_to(trajectory, transition)
            add_to(stats, flatten(info))
            observation = next_observation
        add_to(stats, flatten(info, parent_key="final"))
        trajectories.append(trajectory)

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, trajectories


def evaluate_with_trajectories_and_frames(
    policy_fn,
    env: gym.Env,
    num_episodes: int,
    train_step: int,
    root_dir_name: str = "custom_maze",
    dir_name: str = "custom_maze",
) -> Tuple[Dict[str, float], List[Dict[str, List]]]:
    """
    Same as evaluate, but also returns the trajectories of observations, actions, rewards, etc.

    Arguments:
        See evaluate.
    Returns:
        stats: See evaluate.
        trajectories: A list of dictionaries (each dictionary corresponds to an episode),
            where trajectories[i] = {
                'observation': list_of_observations,
                'action': list_of_actions,
                'next_observation': list_of_next_observations,
                'reward': list of rewards,
                'done': list of done flags,
                'info': list of info dicts,
            }
    """

    trajectories = []
    stats = defaultdict(list)
    save_dir = os.path.join(work_dir, root_dir_name, dir_name)
    for i in range(num_episodes):
        trajectory = defaultdict(list)
        observation, done = env.reset(), False
        visualizer = MazeVisualizer(
            env,
            root_dir=os.path.join(save_dir, f"train_step_{train_step}", f"episode_{i}"),
            frame_dir="frames",
            traj_dir="trajectory",
        )
        step = 0
        while not done:
            visualizer.save_frame(step)
            action = policy_fn(observation)
            next_observation, r, done, info = env.step(action)
            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=r,
                done=done,
                info=info,
            )
            add_to(trajectory, transition)
            add_to(stats, flatten(info))
            observation = next_observation
            step += 1
        add_to(stats, flatten(info, parent_key="final"))
        trajectories.append(trajectory)
        visualizer.draw_traj()
        visualizer.save_traj_data(trajectory)
    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, trajectories


class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        observation = observation["observation"]

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time

            if hasattr(self, "get_normalized_score"):
                info["episode"]["normalized_return"] = self.get_normalized_score(info["episode"]["return"]) * 100.0

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        obs, success = self.env.reset()
        return obs["observation"]
