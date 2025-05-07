import os
import tqdm
from pprint import pprint
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

import jax
import optax
import orbax.checkpoint as orbax
import flax.linen as nn

from jaxrl_m.typing import *
from jaxrl_m.common import TrainState
from jaxrl_m.dataset import Dataset
import jaxrl_m.examples.mujoco.d4rl_utils as d4rl_utils

from adac_model import TransitionModel, ConditionedDiffusion, PredictorMLP, PredictorResNet
from adac_model_util import mish, Normalizer, NormalizerState
from adac_agent_util import msew_loss, l2_loss, var_loss, sample_from_norm

checkpoint_path = os.getcwd()
ckpt_root_dir = "checkpoints"
transition_mlp_ckpt_dir = "transition_mlp"
transition_ddpm_ckpt_dir = "transition_ddpm"
bc_ddpm_ckpt_dir = "bc_ddpm"


class PretrainModelAgent(flax.struct.PyTreeNode):
    model: TrainState
    normalizer: Normalizer
    obs_norm_stat: NormalizerState
    act_norm_stat: NormalizerState


def get_pretrain_model(
    env_name: str,
    key: PRNGKey,
    dataset: Dataset,
    model_type: str,
    # "transition_mlp_00", "transition_mlp_10","transition_mlp_11", "transition_ddpm", "bc_ddpm_0, "bc_ddpm_1"
    use_cache: bool = False,
    checkpoint_path=checkpoint_path,
    ckpt_root_dir=ckpt_root_dir,
):
    save_path = os.path.join(checkpoint_path, ckpt_root_dir, model_type, env_name)

    options = orbax.CheckpointManagerOptions(
        max_to_keep=5,
        best_fn=lambda metric: float(metric),
        best_mode="min",
    )

    checkpoint_manager = orbax.CheckpointManager(
        save_path,
        orbax.PyTreeCheckpointer(),
        options=options,
    )

    best = checkpoint_manager.best_step()
    if use_cache and best is not None:
        if model_type == "transition_mlp_00":
            return load_transition_mlp(checkpoint_manager, dataset, is_stochastic=False)
        elif model_type == "transition_mlp_10" or model_type == "transition_mlp_11":
            return load_transition_mlp(checkpoint_manager, dataset, is_stochastic=True)
        elif model_type == "transition_ddpm":
            return load_transition_ddpm(checkpoint_manager, dataset)
        elif model_type == "bc_ddpm_0":
            return load_bc_ddpm(checkpoint_manager, dataset,use_dl=False)
        elif model_type == "bc_ddpm_1":
            return load_bc_ddpm(checkpoint_manager, dataset,use_dl=True)
        else:
            raise ValueError(f"Invalid model type when using cache: {model_type}.")

    key, subkey = jax.random.split(key)

    if model_type == "transition_mlp_00":
        return train_and_save_transition_mlp(
            checkpoint_manager,
            subkey,
            dataset,
            is_stochastic=False,
            use_logstd=False,
        )
    elif model_type == "transition_mlp_10":
        return train_and_save_transition_mlp(
            checkpoint_manager,
            subkey,
            dataset,
            is_stochastic=True,
            use_logstd=False,
        )
    elif model_type == "transition_mlp_11":
        return train_and_save_transition_mlp(
            checkpoint_manager,
            subkey,
            dataset,
            is_stochastic=True,
            use_logstd=True,
        )
    elif model_type == "transition_ddpm":
        return train_and_save_transition_ddpm(
            checkpoint_manager,
            subkey,
            dataset,
        )
    elif model_type == "bc_ddpm_0":
        return train_and_save_bc_ddpm(
            checkpoint_manager,
            subkey,
            dataset,
            noise_predictor=PredictorMLP,
        )
    elif model_type == "bc_ddpm_1":
        return train_and_save_bc_ddpm(
            checkpoint_manager,
            subkey,
            dataset,
            noise_predictor=PredictorResNet,
        )

    raise ValueError(f"Invalid model type when retraining: {model_type}.")


def train_and_save_transition_mlp(
    ckpt_mngr: orbax.CheckpointManager,
    key: PRNGKey,
    dataset: Dataset,
    *,
    activations: Callable[[Array], Array] = mish,
    split_ratio: float = 0.95,
    is_stochastic: bool = False,
    use_logstd: bool = False,
    hidden_dims: Tuple[int, ...] = (256, 256, 256, 256),
    batch_size: int = 256,
    eval_save_interval: int = 2000,
    max_steps: int = int(3e5),
    lr: float = 3e-4,
    tx: Optional[optax.GradientTransformation] = None,
) -> PretrainModelAgent:
    print("pretrain transition mlp:")
    # ------------------- Data -------------------
    dataset.shuffle()
    train_data, test_data = dataset.split(split_ratio)
    train_dataset = Dataset(train_data)
    test_dataset = Dataset(test_data)

    state_dim = train_data["observations"].shape[-1]
    action_dim = train_data["actions"].shape[-1]

    # ------------------- Model init -------------------
    transition_def = TransitionModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        activations=activations,
        is_stochastic=is_stochastic,
    )
    one_sample = dataset.sample(1)
    key, subkey = jax.random.split(key)
    (out, vars) = transition_def.init_with_output(
        subkey,
        one_sample["observations"],
        one_sample["actions"],
    )
    params = vars["params"]
    pprint(jax.tree.map(lambda x: x.shape, vars))
    pprint(out)

    if tx is None:
        tx = optax.adamw(lr)

    model = TrainState.create(
        transition_def,
        tx=tx,
        params=params,
    )
    # ------------------- Nomalizer -------------------
    normalizer = Normalizer()
    obs_norm_stat = NormalizerState.create_from_data(dataset["observations"])
    obs_norm_stat = normalizer.update_stats(train_data["next_observations"], obs_norm_stat)
    act_norm_stat = NormalizerState.create_from_data(dataset["actions"])

    # ------------------- Training -------------------
    @jax.jit
    def update(
        batch: Batch,
        model: TrainState,
        normalizer: Normalizer,
        obs_norm_stat: NormalizerState,
        act_norm_stat: NormalizerState,
    ) -> Tuple[TrainState, dict]:
        norm_obss = normalizer.normalize(batch["observations"], obs_norm_stat)
        norm_actions = normalizer.normalize(batch["actions"], act_norm_stat)
        diff_gt = batch["next_observations"] - batch["observations"]

        def loss_fn(params):
            pred_diff_means, pred_diff_logstds = model(norm_obss, norm_actions, params=params)
            mloss = msew_loss(pred_diff_means, pred_diff_logstds, diff_gt)
            vloss = var_loss(pred_diff_logstds)
            tot_loss = mloss + 0.2 * vloss
            return tot_loss, {
                "msew_loss": mloss,
                "var_loss": vloss,
                "total_loss": tot_loss,
            }

        new_model, info = model.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
        return new_model, info

    for i in tqdm.tqdm(range(1, max_steps + 1), smoothing=0.1, dynamic_ncols=True):
        batch = train_dataset.sample(batch_size)
        model, info = update(
            batch,
            model,
            normalizer,
            obs_norm_stat,
            act_norm_stat,
        )

        if i % eval_save_interval == 0:
            test_data = test_dataset._dict
            norm_obss = normalizer.normalize(test_data["observations"], obs_norm_stat)
            norm_actions = normalizer.normalize(test_data["actions"], act_norm_stat)
            pred_diffs, pred_diff_logstds = model(norm_obss, norm_actions)
            if use_logstd:
                key, subkey = jax.random.split(key)
                pred_diffs = sample_from_norm(pred_diffs, pred_diff_logstds, subkey)
            preds = test_data["observations"] + pred_diffs
            eval_loss = l2_loss(preds, test_data["next_observations"])
            pretrain_agent = PretrainModelAgent(
                model=model,
                normalizer=normalizer,
                obs_norm_stat=obs_norm_stat,
                act_norm_stat=act_norm_stat,
            )
            ckpt_mngr.save(step=i, items=pretrain_agent, metrics=float(eval_loss))

    best_step = ckpt_mngr.best_step()
    pretrain_agent = ckpt_mngr.restore(best_step, items=pretrain_agent)

    return pretrain_agent


def train_and_save_transition_ddpm(
    ckpt_mngr: orbax.CheckpointManager,
    key: PRNGKey,
    dataset: Dataset,
    *,
    n_timesteps: int = 5,
    max_data: int = 10,
    split_ratio: float = 0.95,
    batch_size: int = 256,
    eval_save_interval: int = 2000,
    max_steps: int = int(3e5),
    lr: float = 3e-4,
    tx: Optional[optax.GradientTransformation] = None,
) -> PretrainModelAgent:
    print("pretrain transition ddpm:")
    dataset.shuffle()
    train_data, test_data = dataset.split(split_ratio)
    train_dataset = Dataset(train_data)
    test_dataset = Dataset(test_data)

    action_dim = train_data["actions"].shape[-1]
    state_dim = train_data["observations"].shape[-1]

    # ------------------- Model init -------------------
    diffusion_def = ConditionedDiffusion(
        data_dim=state_dim,
        condition_dim=state_dim + action_dim,
        n_timesteps=n_timesteps,
        max_data=max_data,
    )
    one_sample = dataset.sample(1)
    key, *subkey = jax.random.split(key, 3)
    (out, vars) = diffusion_def.init_with_output(
        subkey[0],
        jnp.concatenate([one_sample["observations"], one_sample["actions"]], axis=-1),
        subkey[1],
    )
    params = vars["params"]
    pprint(jax.tree.map(lambda x: x.shape, vars))
    pprint(out.shape)

    if tx is None:
        tx = optax.adamw(lr)
    diffusion_state = TrainState.create(
        diffusion_def,
        tx=tx,
        params=params,
    )

    # ------------------- Normalizer -------------------
    normalizer = Normalizer()
    obs_norm_stat = NormalizerState.create_from_data(dataset["observations"])
    obs_norm_stat = normalizer.update_stats(train_data["next_observations"], obs_norm_stat)
    act_norm_stat = NormalizerState.create_from_data(dataset["actions"])

    # ------------------- Training -------------------
    @jax.jit
    def update(
        batch: Batch,
        diffusion_state: TrainState,
        key: PRNGKey,
        normalizer: Normalizer,
        obs_norm_stat: NormalizerState,
        act_norm_stat: NormalizerState,
    ) -> Tuple[TrainState, dict]:
        key, subkey = jax.random.split(key)
        norm_obss = normalizer.normalize(batch["observations"], obs_norm_stat)
        norm_actions = normalizer.normalize(batch["actions"], act_norm_stat)
        norm_next_obss = normalizer.normalize(batch["next_observations"], obs_norm_stat)

        def loss_fn(params):
            loss = diffusion_state(
                norm_next_obss,
                jnp.concatenate([norm_obss, norm_actions], axis=-1),
                rng=subkey,
                params=params,
                method=ConditionedDiffusion.loss,
            )
            return loss, {"training loss": loss}

        new_diffusion_state, info = diffusion_state.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
        return new_diffusion_state, info

    for i in tqdm.tqdm(range(1, max_steps + 1), smoothing=0.1, dynamic_ncols=True):
        key, subkey = jax.random.split(key)
        batch = train_dataset.sample(batch_size)
        diffusion_state, info = update(
            batch,
            diffusion_state,
            subkey,
            normalizer,
            obs_norm_stat,
            act_norm_stat,
        )

        if i % eval_save_interval == 0:
            test_data = test_dataset._dict
            norm_obss = normalizer.normalize(test_data["observations"], obs_norm_stat)
            norm_actions = normalizer.normalize(test_data["actions"], act_norm_stat)
            norm_preds = diffusion_state(jnp.concatenate([norm_obss, norm_actions], axis=-1), subkey)
            preds = normalizer.denormalize(norm_preds, obs_norm_stat)
            eval_loss = l2_loss(preds, test_data["next_observations"])
            pretrain_agent = PretrainModelAgent(
                model=diffusion_state,
                normalizer=normalizer,
                obs_norm_stat=obs_norm_stat,
                act_norm_stat=act_norm_stat,
            )
            ckpt_mngr.save(step=i, items=pretrain_agent, metrics=float(eval_loss))

    best_step = ckpt_mngr.best_step()
    pretrain_agent = ckpt_mngr.restore(best_step, items=pretrain_agent)

    return pretrain_agent


def train_and_save_bc_ddpm(
    ckpt_mngr: orbax.CheckpointManager,
    key: PRNGKey,
    dataset: Dataset,
    *,
    n_timesteps: int = 10,
    max_data: int = 1.0,
    split_ratio: float = 0.95,
    batch_size: int = 256,
    eval_save_interval: int = 2000,
    max_steps: int = int(3e5),
    lr: float = 3e-4,
    tx: Optional[optax.GradientTransformation] = None,
    noise_predictor: nn.Module = PredictorResNet,
) -> PretrainModelAgent:
    print("pretrain bc ddpm:")
    dataset.shuffle()
    train_data, test_data = dataset.split(split_ratio)
    train_dataset = Dataset(train_data)
    test_dataset = Dataset(test_data)

    action_dim = train_data["actions"].shape[-1]
    state_dim = train_data["observations"].shape[-1]

    # ------------------- Model init -------------------
    diffusion_def = ConditionedDiffusion(
        data_dim=action_dim,
        condition_dim=state_dim,
        n_timesteps=n_timesteps,
        max_data=max_data,
        predictor=noise_predictor,
    )

    one_sample = dataset.sample(1)
    key, *subkey = jax.random.split(key, 3)
    (out, vars) = diffusion_def.init_with_output(
        subkey[0],
        one_sample["observations"],
        subkey[1],
    )
    params = vars["params"]
    pprint(jax.tree.map(lambda x: x.shape, vars))
    pprint(out.shape)

    if tx is None:
        tx = optax.adamw(lr)
    diffusion_state = TrainState.create(
        diffusion_def,
        tx=tx,
        params=params,
    )

    # ------------------- Normalizer -------------------
    normalizer = Normalizer()
    obs_norm_stat = NormalizerState.create_from_data(dataset["observations"])
    obs_norm_stat = normalizer.update_stats(train_data["next_observations"], obs_norm_stat)
    act_norm_stat = NormalizerState.create_from_data(dataset["actions"])

    # ------------------- Training -------------------
    @jax.jit
    def update(
        batch: Batch,
        diffusion_state: TrainState,
        key: PRNGKey,
        normalizer: Normalizer,
        obs_norm_stat: NormalizerState,
        act_norm_stat: NormalizerState,
    ) -> Tuple[TrainState, dict]:
        key, subkey = jax.random.split(key)
        norm_obss = normalizer.normalize(batch["observations"], obs_norm_stat)
        norm_actions = normalizer.normalize(batch["actions"], act_norm_stat)

        def loss_fn(params):
            loss = diffusion_state(
                norm_actions,
                norm_obss,
                rng=subkey,
                params=params,
                method=ConditionedDiffusion.loss,
            )
            return loss, {"training loss": loss}

        new_diffusion_state, info = diffusion_state.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
        return new_diffusion_state, info

    for i in tqdm.tqdm(range(1, max_steps + 1), smoothing=0.1, dynamic_ncols=True):
        key, subkey = jax.random.split(key)
        batch = train_dataset.sample(batch_size)
        diffusion_state, info = update(
            batch,
            diffusion_state,
            subkey,
            normalizer,
            obs_norm_stat,
            act_norm_stat,
        )

        if i % eval_save_interval == 0:
            test_data = test_dataset._dict
            norm_obss = normalizer.normalize(test_data["observations"], obs_norm_stat)
            norm_preds = diffusion_state(norm_obss, subkey)
            preds = normalizer.denormalize(norm_preds, act_norm_stat)
            eval_loss = l2_loss(preds, test_data["actions"])
            pretrain_agent = PretrainModelAgent(
                model=diffusion_state,
                normalizer=normalizer,
                obs_norm_stat=obs_norm_stat,
                act_norm_stat=act_norm_stat,
            )
            ckpt_mngr.save(step=i, items=pretrain_agent, metrics=float(eval_loss))

    best_step = ckpt_mngr.best_step()
    pretrain_agent = ckpt_mngr.restore(best_step, items=pretrain_agent)

    return pretrain_agent


def load_transition_mlp(
    ckpt_mngr: orbax.CheckpointManager,
    dataset: Dataset,
    *,
    activations: Callable[[Array], Array] = mish,
    is_stochastic: bool = False,
    hidden_dims: Tuple[int, ...] = (256, 256, 256, 256),
    lr: float = 3e-4,
    tx: Optional[optax.GradientTransformation] = None,
) -> PretrainModelAgent:
    if tx is None:
        tx = optax.adamw(lr)
    # 构造 dummy 模型，用于恢复时匹配保存时的结构
    state_dim = dataset["observations"].shape[-1]
    action_dim = dataset["actions"].shape[-1]
    transition_def = TransitionModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        activations=activations,
        is_stochastic=is_stochastic,
    )
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    # 使用 init_with_output 初始化，获得一个完整的参数结构
    one_sample = dataset.sample(1)
    dummy_vars = transition_def.init(subkey, one_sample["observations"], one_sample["actions"])
    dummy_params = dummy_vars["params"]
    dummy_state = TrainState.create(
        transition_def,
        tx=tx,
        params=dummy_params,
    )
    # 构造 dummy 模板：注意 normalizer 和 NormalizerState 的初始化也要和训练时一致
    dummy_agent = PretrainModelAgent(
        model=dummy_state,
        normalizer=Normalizer(),
        obs_norm_stat=NormalizerState.create_from_data(dataset["observations"]),
        act_norm_stat=NormalizerState.create_from_data(dataset["actions"]),
    )
    # 使用 dummy_agent 作为模板恢复 checkpoint 中保存的对象
    restored_agent = ckpt_mngr.restore(ckpt_mngr.best_step(), items=dummy_agent)
    return restored_agent


def load_transition_ddpm(
    ckpt_mngr: orbax.CheckpointManager,
    dataset: Dataset,
    *,
    n_timesteps: int = 5,
    max_data: int = 10,
    lr: float = 3e-4,
    tx: Optional[optax.GradientTransformation] = None,
) -> PretrainModelAgent:
    if tx is None:
        tx = optax.adamw(lr)
    state_dim = dataset["observations"].shape[-1]
    action_dim = dataset["actions"].shape[-1]

    diffusion_def = ConditionedDiffusion(
        data_dim=state_dim,
        condition_dim=state_dim + action_dim,
        n_timesteps=n_timesteps,
        max_data=max_data,
    )
    one_sample = dataset.sample(1)
    key = jax.random.PRNGKey(0)
    key, *subkey = jax.random.split(key, 3)
    dummy_vars = diffusion_def.init(
        subkey[0],
        jnp.concatenate([one_sample["observations"], one_sample["actions"]], axis=-1),
        subkey[1],
    )
    dummy_params = dummy_vars["params"]
    dummy_state = TrainState.create(
        diffusion_def,
        tx=tx,
        params=dummy_params,
    )
    dummy_agent = PretrainModelAgent(
        model=dummy_state,
        normalizer=Normalizer(),
        obs_norm_stat=NormalizerState.create_from_data(dataset["observations"]),
        act_norm_stat=NormalizerState.create_from_data(dataset["actions"]),
    )
    restored_agent = ckpt_mngr.restore(ckpt_mngr.best_step(), items=dummy_agent)
    return restored_agent


def load_bc_ddpm(
    ckpt_mngr: orbax.CheckpointManager,
    dataset: Dataset,
    *,
    use_dl: bool = False,
    n_timesteps: int = 5,
    max_data: int = 10,
    lr: float = 3e-4,
    tx: Optional[optax.GradientTransformation] = None,
) -> PretrainModelAgent:
    if tx is None:
        tx = optax.adamw(lr)
    state_dim = dataset["observations"].shape[-1]
    action_dim = dataset["actions"].shape[-1]

    diffusion_def = ConditionedDiffusion(
        data_dim=action_dim,
        condition_dim=state_dim,
        n_timesteps=n_timesteps,
        max_data=max_data,
        predictor=PredictorResNet if use_dl else PredictorMLP,
    )

    one_sample = dataset.sample(1)
    key = jax.random.PRNGKey(0)
    key, *subkey = jax.random.split(key, 3)
    dummy_vars = diffusion_def.init(
        subkey[0],
        one_sample["observations"],
        subkey[1],
    )
    dummy_params = dummy_vars["params"]
    dummy_state = TrainState.create(
        diffusion_def,
        tx=tx,
        params=dummy_params,
    )
    dummy_agent = PretrainModelAgent(
        model=dummy_state,
        normalizer=Normalizer(),
        obs_norm_stat=NormalizerState.create_from_data(dataset["observations"]),
        act_norm_stat=NormalizerState.create_from_data(dataset["actions"]),
    )
    restored_agent = ckpt_mngr.restore(ckpt_mngr.best_step(), items=dummy_agent)
    return restored_agent


def get_and_check_transition_and_bc(
    key: PRNGKey,
    env_name: str = "halfcheetah-expert-v2",
    dataset: Dataset = None,
    transition_model_type: str = "transition_mlp_00",
    bc_model_type: str = "bc_ddpm_0",
    use_transition_cache: bool = False,
    use_bc_cache: bool = False,
    checkpoint_path=checkpoint_path,
    ckpt_root_dir=ckpt_root_dir,
) -> Tuple[PretrainModelAgent, PretrainModelAgent]:
    _, key = jax.random.split(key)

    if dataset is None:
        env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env)

    transition_agent, bc_agent = None, None

    if "mlp" in transition_model_type:
        transition_agent = get_pretrain_model(
            env_name,
            key,
            dataset,
            transition_model_type,
            use_cache=use_transition_cache,
            checkpoint_path=checkpoint_path,
            ckpt_root_dir=ckpt_root_dir,
        )
        data = dataset.sample(10)
        norm_obss = transition_agent.normalizer.normalize(data["observations"], transition_agent.obs_norm_stat)
        norm_actions = transition_agent.normalizer.normalize(data["actions"], transition_agent.act_norm_stat)
        pred_diffs, _ = transition_agent.model(norm_obss, norm_actions)
        preds = data["observations"] + pred_diffs
        print(l2_loss(preds, data["next_observations"]))
    elif transition_model_type == "transition_ddpm":
        transition_agent = get_pretrain_model(
            env_name,
            key,
            dataset,
            transition_model_type,
            use_cache=use_transition_cache,
            checkpoint_path=checkpoint_path,
            ckpt_root_dir=ckpt_root_dir,
        )
        data = dataset.sample(10)
        norm_obss = transition_agent.normalizer.normalize(data["observations"], transition_agent.obs_norm_stat)
        norm_actions = transition_agent.normalizer.normalize(data["actions"], transition_agent.act_norm_stat)
        key, subkey = jax.random.split(key)
        norm_preds = transition_agent.model(jnp.concatenate([norm_obss, norm_actions], axis=-1), subkey)
        preds = transition_agent.normalizer.denormalize(norm_preds, transition_agent.obs_norm_stat)
        print(l2_loss(preds, data["next_observations"]))
    else:
        raise ValueError(f"Invalid transition model type: {transition_model_type}.")

    if "bc_ddpm" in bc_model_type:
        bc_agent = get_pretrain_model(
            env_name,
            key,
            dataset,
            bc_model_type,
            use_cache=use_bc_cache,
            checkpoint_path=checkpoint_path,
            ckpt_root_dir=ckpt_root_dir,
        )
        data = dataset.sample(10)
        norm_obss = bc_agent.normalizer.normalize(data["observations"], bc_agent.obs_norm_stat)
        key, subkey = jax.random.split(key)
        norm_preds = bc_agent.model(norm_obss, subkey)
        preds = bc_agent.normalizer.denormalize(norm_preds, bc_agent.act_norm_stat)
        print(l2_loss(preds, data["actions"]))
    else:
        raise ValueError(f"Invalid bc model type: {bc_model_type}.")

    return transition_agent, bc_agent


def main():
    key = jax.random.PRNGKey(0)
    env_name = "halfcheetah-expert-v2"
    env = d4rl_utils.make_env(env_name)
    dataset = d4rl_utils.get_dataset(env)
    model_type = "bc_ddpm_1"

    if "mlp" in model_type:
        pretrain_agent = get_pretrain_model(env_name, key, dataset, model_type, use_cache=False)
        pretrain_agent = get_pretrain_model(env_name, key, dataset, model_type, use_cache=True)
        data = dataset.sample(10)
        norm_obss = pretrain_agent.normalizer.normalize(data["observations"], pretrain_agent.obs_norm_stat)
        norm_actions = pretrain_agent.normalizer.normalize(data["actions"], pretrain_agent.act_norm_stat)
        pred_diffs, _ = pretrain_agent.model(norm_obss, norm_actions)
        preds = data["observations"] + pred_diffs
        print(l2_loss(preds, data["next_observations"]))
    elif model_type == "transition_ddpm":
        pretrain_agent = get_pretrain_model(env_name, key, dataset, model_type, use_cache=False)
        pretrain_agent = get_pretrain_model(env_name, key, dataset, model_type, use_cache=True)
        data = dataset.sample(10)
        norm_obss = pretrain_agent.normalizer.normalize(data["observations"], pretrain_agent.obs_norm_stat)
        norm_actions = pretrain_agent.normalizer.normalize(data["actions"], pretrain_agent.act_norm_stat)
        key, subkey = jax.random.split(key)
        norm_preds = pretrain_agent.model(jnp.concatenate([norm_obss, norm_actions], axis=-1), subkey)
        preds = pretrain_agent.normalizer.denormalize(norm_preds, pretrain_agent.obs_norm_stat)
        print(l2_loss(preds, data["next_observations"]))
    elif "bc_ddpm" in model_type:
        pretrain_agent = get_pretrain_model(env_name, key, dataset, model_type, use_cache=False)
        pretrain_agent = get_pretrain_model(env_name, key, dataset, model_type, use_cache=True)
        data = dataset.sample(10)
        norm_obss = pretrain_agent.normalizer.normalize(data["observations"], pretrain_agent.obs_norm_stat)
        key, subkey = jax.random.split(key)
        norm_preds = pretrain_agent.model(norm_obss, subkey)
        preds = pretrain_agent.normalizer.denormalize(norm_preds, pretrain_agent.act_norm_stat)
        print(l2_loss(preds, data["actions"]))


if __name__ == "__main__":
    # _, _ = get_and_check_transition_and_bc(
    #     jax.random.PRNGKey(0), transition_model_type="transition_ddpm", use_transition_cache=False, use_bc_cache=True
    # )

    main()
