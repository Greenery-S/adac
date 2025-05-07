import jax
import jax.numpy as jnp
import optax
import flax
from flax import linen as nn
from jaxrl_m.common import target_update, TrainState
from jaxrl_m.typing import PRNGKey, Batch

from adac_model import *
from adac_pretrain import get_and_check_transition_and_bc, PretrainModelAgent
from adac_agent_util import expectile_loss, soft_clip,tanh_scale

from jax import vmap
from jax.lax import stop_gradient
from jax.random import uniform
from jaxrl_m.networks import ValueCritic, Critic, ensemblize
from typing import Sequence, Optional, Tuple
from pprint import pprint
import distrax
from pprint import pprint


class ABACAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    actor: TrainState
    ema_actor: TrainState  # 和target_critic的更新方式一样
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    bc_actor: PretrainModelAgent
    transition: PretrainModelAgent
    config: dict = flax.struct.field(pytree_node=False)
    # 是否是树节点, 类似于torch的register_buffer
    step: int = 0

    @jax.jit
    def update(agent, batch: Batch):

        def value_loss_fn(value_params):
            q1, q2 = agent.target_critic(batch["observations"], batch["actions"])
            q = jnp.minimum(q1, q2)
            v = agent.value(batch["observations"], params=value_params)
            value_loss = expectile_loss(q - v, agent.config["expectile"]).mean()
            return value_loss, {
                "value_loss": value_loss,
                "v": v.mean(),
            }

        def critic_loss_fn(critic_params):
            new_rng, maxq_key, adv_bc_key, adv_bc_trans_key, adv_actor_trans_key = jax.random.split(agent.rng, 5)
            q1, q2 = agent.critic(batch["observations"], batch["actions"], params=critic_params)

            if agent.config["is_max_q_backup"]:

                def maxq_backup_fn(one_state, key) -> Tuple[jnp.ndarray, jnp.ndarray]:
                    """handle 1 state
                    1. repeat state
                    2. get repeated action
                    3. get repeated q values from q1 and q2
                    4. get max q values and its actions in q1 and q2 individually
                    5. get final q values and its actions by min(q1, q2)
                    """
                    repeats = agent.config["max_q_repeat"]
                    state_rpt = jnp.repeat(one_state[None, ...], repeats, axis=0)
                    action_rpt = agent.ema_actor(state_rpt, key)
                    q1, q2 = agent.target_critic(state_rpt, action_rpt)  # (R, )
                    q1_max = jnp.max(q1)
                    q2_max = jnp.max(q2)
                    q_nxt = jnp.minimum(q1_max, q2_max)
                    # action corresponding to q_max
                    q1_argmax = jnp.argmax(q1)
                    q2_argmax = jnp.argmax(q2)
                    action_nxt = jnp.where(q1_max <= q2_max, action_rpt[q1_argmax], action_rpt[q2_argmax])
                    return q_nxt, action_nxt

                q_nxt, action_nxt = vmap(maxq_backup_fn, in_axes=(0, None))(batch["next_observations"], maxq_key)
            else:
                action_nxt = agent.ema_actor(batch["next_observations"], maxq_key)
                q1_nxt, q2_nxt = agent.target_critic(batch["next_observations"], action_nxt)
                q_nxt = jnp.minimum(q1_nxt, q2_nxt)

            # 计算 adv
            if agent.config["neg_lbd"] or agent.config["pos_lbd"]:
                repeats, transition_model_type = agent.config["adv_repeat"], agent.config["transition_model_type"]
                next_state_rpt = jnp.repeat(batch["next_observations"], repeats, axis=0)
                # get bc actions
                norm_next_state_rpt = agent.bc_actor.normalizer.normalize(next_state_rpt, agent.bc_actor.obs_norm_stat)
                norm_next_bc_actions = agent.bc_actor.model(norm_next_state_rpt, adv_bc_key)
                next_bc_actions = agent.bc_actor.normalizer.denormalize(norm_next_bc_actions, agent.bc_actor.act_norm_stat)
                # get actor actions
                next_actor_actions = action_nxt  # (B, A)
                # get s'' for both bc and actor
                norm_next_state = agent.transition.normalizer.normalize(batch["next_observations"], agent.transition.obs_norm_stat)
                norm_next_state_rpt = agent.transition.normalizer.normalize(next_state_rpt, agent.transition.obs_norm_stat)
                norm_next_bc_actions = agent.transition.normalizer.normalize(next_bc_actions, agent.transition.act_norm_stat)
                norm_next_actor_actions = agent.transition.normalizer.normalize(next_actor_actions, agent.transition.act_norm_stat)
                if "mlp" in transition_model_type:
                    pred_bc_diffs, _ = agent.transition.model(norm_next_state_rpt, norm_next_bc_actions)
                    nnext_bc_states = next_state_rpt + pred_bc_diffs  # (BxR, A)
                    pred_actor_diffs, _ = agent.transition.model(norm_next_state, norm_next_actor_actions)
                    nnext_actor_states = batch["next_observations"] + pred_actor_diffs  # (B, A)
                elif "ddpm" in transition_model_type:
                    norm_bc_preds = agent.transition.model(jnp.concatenate([norm_next_state_rpt, norm_next_bc_actions], axis=-1), adv_bc_trans_key)
                    nnext_bc_states = agent.transition.normalizer.denormalize(norm_bc_preds, agent.transition.obs_norm_stat)  # (BxR, A)
                    norm_actor_preds = agent.transition.model(
                        jnp.concatenate([norm_next_state, norm_next_actor_actions], axis=-1), adv_actor_trans_key
                    )
                    nnext_actor_states = agent.transition.normalizer.denormalize(norm_actor_preds, agent.transition.obs_norm_stat)  # (B, A)
                else:
                    raise NotImplementedError
                v_nnxt_bc = agent.value(nnext_bc_states).reshape(-1, repeats)  # (batch_size, repeats)
                v_nnxt_actor = agent.value(nnext_actor_states)  # (batch_size, )
                bc_quantile = agent.config["bc_quantile"]
                adv = v_nnxt_actor - jnp.quantile(v_nnxt_bc, bc_quantile, axis=-1)  #
                adv=tanh_scale(adv,agent.config["neg_lbd"],agent.config["pos_lbd"])
                # adv = jnp.where(adv < 0, adv * agent.config["neg_lbd"], adv * agent.config["pos_lbd"])
                # adv = jnp.clip(adv, -10, 15)  # TODO: 限制adv的范围
            else:
                adv = jnp.zeros_like(q_nxt)

            # adv = jnp.where(agent.config["warmup_steps"] * 30 > agent.step, 0, adv)
            target_q = stop_gradient(batch["rewards"] + agent.config["discount"] * batch["masks"] * q_nxt + adv)
            critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

            positive_adv = jnp.clip(adv, 0, None)
            negative_adv = jnp.clip(adv, None, 0)
            return critic_loss, {
                "adv_positive_mean": jnp.sum(positive_adv) / jnp.sum(adv > 0),
                "adv_positive_ratio": jnp.sum(adv > 0) / adv.size,
                "adv_negative_mean": jnp.sum(negative_adv) / jnp.sum(adv < 0),
                "adv_negative_ratio": jnp.sum(adv < 0) / adv.size,
                "critic_loss": critic_loss,
                "q1_mean": q1.mean(),
                "q2_mean": q2.mean(),
                "q1_std": q1.std(),
                "q2_std": q2.std(),
                "new_rng": new_rng,
            }

        def actor_loss_fn(actor_params):
            new_rng, *subkey = jax.random.split(agent.rng, 4)
            bc_loss = agent.actor(
                batch["actions"],
                batch["observations"],
                subkey[0],
                params=actor_params,
                method=ConditionedDiffusion.loss,
            )
            action_nxt = agent.actor(batch["observations"], subkey[1])
            q1_nxt, q2_nxt = agent.critic(batch["observations"], action_nxt)
            if agent.config["use_auto_norm"]:
                q_loss = jnp.where(
                    uniform(subkey[2]) < 0.5,
                    -q1_nxt.mean() / stop_gradient(jnp.abs(q2_nxt).mean()),
                    -q2_nxt.mean() / stop_gradient(jnp.abs(q1_nxt).mean()),
                )
            else:
                q_loss = -(q1_nxt.mean() + q2_nxt.mean())/2
            actor_loss = bc_loss + q_loss * agent.config["eta"]
            return actor_loss, {
                "actor_loss": actor_loss,
                "bc_loss": bc_loss,
                "q_loss": q_loss * agent.config["eta"],
                "new_rng": new_rng,
            }

        new_value, value_info = agent.value.apply_loss_fn(loss_fn=value_loss_fn, has_aux=True)
        new_critic, critic_info = agent.critic.apply_loss_fn(loss_fn=critic_loss_fn, has_aux=True)
        agent = agent.replace(rng=critic_info["new_rng"])
        new_actor, actor_info = agent.actor.apply_loss_fn(loss_fn=actor_loss_fn, has_aux=True)
        agent = agent.replace(rng=actor_info["new_rng"])

        new_target_critic = target_update(agent.critic, agent.target_critic, agent.config["target_update_rate"])

        condition = jnp.logical_and(
            agent.step >= agent.config["warmup_steps"],
            jnp.equal(jnp.mod(agent.step + 1, agent.config["ema_update_interval"]), 0),
        )
        new_ema_actor = jax.lax.cond(
            condition,
            lambda _: target_update(agent.actor, agent.ema_actor, agent.config["ema_update_rate"]),
            lambda _: agent.ema_actor,
            operand=None,
        )

        return agent.replace(
            value=new_value,
            critic=new_critic,
            target_critic=new_target_critic,
            actor=new_actor,
            ema_actor=new_ema_actor,
            step=agent.step + 1,
        ), {**critic_info, **actor_info, **value_info}

    @jax.jit
    def sample_actions(
        agent,
        obs: jnp.ndarray,
        *,
        seed: PRNGKey,
        temperature: float = 1.0,
        # 用于控制 softmax 的温度, 越小越接近 argmax
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # 重复 obs
        obs = jnp.asarray(obs)[None, ...]
        obs_rpt = jnp.repeat(obs, repeats=agent.config["num_samples"], axis=0)

        # 批量采样候选动作
        action_rpt = agent.actor(obs_rpt, rng=seed, method=ConditionedDiffusion.sample)

        # 计算 q 值
        q1, q2 = agent.target_critic(obs_rpt, action_rpt)
        q_min = jnp.minimum(q1, q2).reshape(-1)

        # 用 distrax 的 Categorical 分布，从 logits 做一次采样
        # distrax.Categorical(logits=...)会等效于 softmax(...)
        dist = distrax.Categorical(logits=q_min / temperature)
        seed, subkey = jax.random.split(seed)
        idx = dist.sample(seed=subkey)  # 返回 [0..49] 间的一个索引

        return action_rpt[idx]


def create_learner(
    seed: int,
    env_name: str,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    max_action: float = 1.0,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    hidden_dims: Sequence[int] = (256, 256, 256),
    discount: float = 0.99,
    target_update_rate: float = 5e-3,
    ema_update_rate: float = 5e-3,
    warmup_steps: int = 5000,
    ema_update_interval: int = 5,
    eta: float = 1.0,
    is_max_q_backup: bool = False,
    max_q_repeat: int = 10,
    beta_schedule: str = "vp",
    loss_type: str = "l2",
    n_timesteps: int = 100,
    is_critic_opt_decay: bool = False,
    is_actor_opt_decay: bool = False,
    opt_max_steps: int = int(2e6),
    grad_norm_clip: float = 5.0,
    num_samples: int = 50,
    transition_model_type: str = "transition_mlp_00",
    bc_quantile: float = 0.75,
    adv_repeat: int = 100,
    expectile: float = 0.8,
    # lbd: float = 1.0,  # lambda
    neg_lbd: float = 1.0,
    pos_lbd: float = 1.0,
    use_auto_norm: bool = True,
    use_transition_cache: bool = False,
    use_bc_cache: bool = False,
    use_dl: bool = False,
    **kwargs,
):
    key = jax.random.PRNGKey(seed)
    action_dim = actions.shape[-1]
    state_dim = observations.shape[-1]
    diffusion_def = ConditionedDiffusion(
        action_dim,
        state_dim,
        max_data=max_action,
        beta_schedule=beta_schedule,
        n_timesteps=n_timesteps,
        loss_type=loss_type,
        # predictor= PredictorResNet if use_dl else PredictorMLP,
    )

    ##############################
    # DIFFUSION ACTOR
    ##############################
    if is_actor_opt_decay:
        schedule_fn = optax.cosine_decay_schedule(-actor_lr, opt_max_steps)
        actor_tx = optax.chain(
            optax.clip_by_global_norm(grad_norm_clip),
            optax.scale_by_adam(),
            optax.scale_by_schedule(schedule_fn),
        )
    else:
        actor_tx = optax.chain(
            optax.clip_by_global_norm(grad_norm_clip),
            optax.adam(learning_rate=actor_lr),
        )
    key, *subkey = jax.random.split(key, 3)
    (out, actor_var) = diffusion_def.init_with_output(
        subkey[0],
        observations,
        subkey[1],
    )
    actor_params = actor_var["params"]

    pprint(jax.tree.map(lambda x: x.shape, actor_params))
    pprint(out)
    diffusion_actor = TrainState.create(
        diffusion_def,
        tx=actor_tx,
        params=actor_params,
    )
    ema_actor = TrainState.create(diffusion_def, params=actor_params)

    ##############################
    # DOUBLE Q CRITIC
    ##############################
    key, subkey = jax.random.split(key)
    critic_def = ensemblize(Critic, num_qs=2)(hidden_dims, mish) if not use_dl else ensemblize(CriticResNet, num_qs=2)()
    if is_critic_opt_decay:
        schedule_fn = optax.cosine_decay_schedule(-critic_lr, opt_max_steps)
        critic_tx = optax.chain(
            optax.clip_by_global_norm(grad_norm_clip),
            optax.scale_by_adam(),
            optax.scale_by_schedule(schedule_fn),
        )
    else:
        critic_tx = optax.chain(
            optax.clip_by_global_norm(grad_norm_clip),
            optax.adam(learning_rate=critic_lr),
        )
    critic_params = critic_def.init(subkey, observations, actions)["params"]
    critic = TrainState.create(
        critic_def,
        critic_params,
        tx=critic_tx,
    )
    target_critic = TrainState.create(critic_def, critic_params)

    ##############################
    # VALUE CRITIC
    ##############################
    key, subkey = jax.random.split(key)
    value_def = ValueCritic(hidden_dims) if not use_dl else ValueCriticResNet()
    if is_critic_opt_decay:
        schedule_fn = optax.cosine_decay_schedule(-critic_lr, opt_max_steps)
        value_tx = optax.chain(
            optax.clip_by_global_norm(grad_norm_clip),
            optax.scale_by_adam(),
            optax.scale_by_schedule(schedule_fn),
        )
    else:
        value_tx = optax.chain(
            optax.clip_by_global_norm(grad_norm_clip),
            optax.adam(learning_rate=critic_lr),
        )
    value_params = value_def.init(subkey, observations)["params"]
    value = TrainState.create(
        value_def,
        value_params,
        tx=value_tx,
    )

    ##############################
    # PretrainModelAgents
    ##############################
    key, subkey = jax.random.split(key)
    transition, bc_actor = get_and_check_transition_and_bc(
        subkey,
        env_name,
        None,
        use_bc_cache=use_bc_cache,
        use_transition_cache=use_transition_cache,
        transition_model_type=transition_model_type,
        bc_model_type="bc_ddpm_0" ,
    )

    config = flax.core.FrozenDict(
        dict(
            discount=discount,
            target_update_rate=target_update_rate,
            ema_update_rate=ema_update_rate,
            warmup_steps=warmup_steps,
            ema_update_interval=ema_update_interval,
            eta=eta,
            is_max_q_backup=is_max_q_backup,
            max_q_repeat=max_q_repeat,
            num_samples=num_samples,
            adv_repeat=adv_repeat,
            expectile=expectile,
            bc_quantile=bc_quantile,
            transition_model_type=transition_model_type,
            # lbd=lbd,
            use_auto_norm=use_auto_norm,
            neg_lbd=neg_lbd,
            pos_lbd=pos_lbd,
        )
    )

    return ABACAgent(
        rng=key,
        value=value,
        actor=diffusion_actor,
        ema_actor=ema_actor,
        critic=critic,
        target_critic=target_critic,
        transition=transition,
        bc_actor=bc_actor,
        config=config,
    )
