import os, pickle, yaml, tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from absl import app, flags
from functools import partial
from dataclasses import asdict

import wandb
from numpy.random import choice
from jax import random
from ml_collections import config_flags
from flax.training import checkpoints

import jaxrl_m.examples.mujoco.d4rl_utils as d4rl_utils
from jaxrl_m.wandb import setup_wandb, default_wandb_config, get_flag_dict
from jaxrl_m.evaluation import supply_rng, evaluate

import adac_agent as learner
from adac_run_util import get_tuned_dataset, get_name_suffix, TopKHeap, convert_checkpoint_dict
from adac_hyper import hyperparameters, get_default_config


FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "antmaze-umaze-v0", "Environment name.")
flags.DEFINE_string("save_dir", None, "Logging dir (if not None, save params).")
flags.DEFINE_integer("seed", choice(1000000), "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("save_interval", 50000, "Save interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(2e6), "Number of training steps.")

wandb_config = default_wandb_config()
wandb_config.update(
    {
        "project": "d4rl_test",
        "group": "dql_test",
        "name": "abac_{env_name}",
    }
)

config_flags.DEFINE_config_dict("wandb", wandb_config, lock_config=False)
config_flags.DEFINE_config_dict("config", get_default_config(), lock_config=False)

overlay = {
    ########################################
    ###Put your test hyperparameters here###
    ########################################
    # "use_auto_norm": True,
    # "is_max_q_backup": True,
    # "max_q_repeat": 5,
    # "n_timesteps": 10,
    # "expectile": 0.68,
    # "bc_quantile": 0.83,
    # "use_dl": True,
}


def main(_):
    topk_manager_offline = TopKHeap(k=50, ms_type="offline")
    topk_manager_online = TopKHeap(k=50, ms_type="online")
    ckpt_file_name = f"topk_checkpoints{get_name_suffix(use_time=True)}.yaml"

    env_name = FLAGS.env_name
    env_cfg = hyperparameters.get(env_name, {})
    FLAGS.config.update(env_cfg)
    FLAGS.config.update(overlay)

    # 创建 wandb 日志记录器
    base_name = FLAGS.wandb["name"].format(env_name=env_name)
    FLAGS.wandb["name"] = base_name + get_name_suffix(use_time=True)
    setup_wandb(FLAGS.config.to_dict(), **FLAGS.wandb)

    if FLAGS.save_dir is not None:
        FLAGS.save_dir = os.path.join(
            FLAGS.save_dir,
            wandb.run.project,
            wandb.config.exp_prefix,
            wandb.config.experiment_id,
        )
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        print(f"Saving config to {FLAGS.save_dir}/config.pkl")
        with open(os.path.join(FLAGS.save_dir, "config.pkl"), "wb") as f:
            pickle.dump(get_flag_dict(), f)

    env = d4rl_utils.make_env(env_name)
    dataset = d4rl_utils.get_dataset(env)
    reward_tune = FLAGS.config.reward_tune
    dataset = get_tuned_dataset(dataset, reward_tune)

    example_batch = dataset.sample(1)
    agent = learner.create_learner(
        FLAGS.seed,
        env_name,
        example_batch["observations"],
        example_batch["actions"],
        max_action=env.action_space.high[0],
        **FLAGS.config,
    )
    temperature = FLAGS.config.get("temperature", 1.0)

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, dynamic_ncols=True):

        batch = dataset.sample(FLAGS.batch_size)
        agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)

        if i % FLAGS.eval_interval == 0:
            policy_fn = partial(
                supply_rng(agent.sample_actions),
                temperature=temperature,
            )
            eval_info = evaluate(policy_fn, env, num_episodes=FLAGS.eval_episodes)
            eval_metrics = {f"evaluation/{k}": v for k, v in eval_info.items()}
            wandb.log(eval_metrics, step=i)

            bc_loss = update_info["bc_loss"]
            eval_score = eval_info["final.episode.normalized_return"]
            topk_manager_offline.add(bc_loss=bc_loss, eval_score=eval_score, step=i)
            topk_manager_online.add(bc_loss=bc_loss, eval_score=eval_score, step=i)

            # save topk checkpoints
            data_to_save = {
                "offline": [convert_checkpoint_dict(asdict(cp)) for cp in topk_manager_offline.get_all()],
                "online": [convert_checkpoint_dict(asdict(cp)) for cp in topk_manager_online.get_all()],
            }
            with open(ckpt_file_name, "w") as f:
                yaml.dump(data_to_save, f, allow_unicode=True, sort_keys=False)
            wandb.save(ckpt_file_name)

        if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None:
            checkpoints.save_checkpoint(FLAGS.save_dir, agent, i)


if __name__ == "__main__":
    app.run(main)
