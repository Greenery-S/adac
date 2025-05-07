import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Any, Optional
from adac_model_util import (
    extract,
    linear_beta_schedule,
    cosine_beta_schedule,
    vp_beta_schedule,
    SinusoidalPosEmb,
    mish,
)
from jaxrl_m.typing import *
from jaxrl_m.networks import MLP
from functools import partial
from flax.linen.initializers import variance_scaling


##############################################
# residual block
##############################################

lecun_unfirom = variance_scaling(1 / 3, "fan_in", "uniform")
bias_init = nn.initializers.zeros


class ResidualBlock(nn.Module):
    width: int
    norm_cls: callable = nn.LayerNorm
    activation: callable = mish

    @nn.compact
    def __call__(self, x):
        identity = x
        x = nn.Dense(self.width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = self.norm_cls()(x)
        x = self.activation(x)
        x = nn.Dense(self.width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = self.norm_cls()(x)
        x = self.activation(x)
        x = nn.Dense(self.width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = self.norm_cls()(x)
        x = self.activation(x)
        x = nn.Dense(self.width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = self.norm_cls()(x)
        x = self.activation(x)
        return x + identity


##############################################
# deep critics
##############################################


class CriticResNet(nn.Module):
    width: int = 256
    depth: int = 16
    activations: Callable = mish

    def setup(self):
        num_blocks = self.depth // 4
        self.blocks = [ResidualBlock(width=self.width, norm_cls=nn.LayerNorm, activation=self.activations) for _ in range(num_blocks)]

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        inputs = nn.Dense(self.width)(inputs)
        for block in self.blocks:
            inputs = block(inputs)
        critic = nn.Dense(1)(inputs)
        return jnp.squeeze(critic, -1)


class ValueCriticResNet(nn.Module):
    width: int = 256
    depth: int = 16
    activations: Callable = mish

    def setup(self):
        num_blocks = self.depth // 4
        self.blocks = [ResidualBlock(width=self.width, norm_cls=nn.LayerNorm, activation=self.activations) for _ in range(num_blocks)]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        inputs = nn.Dense(self.width)(observations)
        for block in self.blocks:
            inputs = block(inputs)
        critic = nn.Dense(1)(inputs)
        return jnp.squeeze(critic, -1)


##############################################
# transition model (s,a -> s')
##############################################


class TransitionModel(nn.Module):
    state_dim: int
    action_dim: int
    hidden_dims: Sequence = (256, 256)
    activations: Callable = mish
    is_stochastic: bool = False
    max_logstd: float = 0.5
    min_logstd: float = -5.0

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([state, action], -1)
        diff_mean = MLP((*self.hidden_dims, self.state_dim), activations=self.activations)(inputs)
        diff_logstd = jnp.zeros_like(diff_mean)
        if self.is_stochastic:
            diff_logstd = MLP((*self.hidden_dims, self.state_dim), activations=self.activations)(inputs)
            diff_logstd = jnp.clip(diff_logstd, self.min_logstd, self.max_logstd)
        return diff_mean, diff_logstd


##############################################
# noise model
##############################################


class PredictorMLP(nn.Module):
    data_dim: int  # 生成数据的维度
    condition_dim: int  # 条件的维度
    t_dim: int = 16

    def setup(self):
        self.time_emb = nn.Sequential(
            [
                SinusoidalPosEmb(self.t_dim),
                nn.Dense(self.t_dim * 2),
                mish,
                nn.Dense(self.t_dim),
            ]
        )

        self.net = nn.Sequential(
            [
                nn.Dense(256),
                mish,
                nn.Dense(256),
                mish,
                nn.Dense(256),
                mish,
                nn.Dense(256),
                mish,
                nn.Dense(256),
                mish,
                nn.Dense(self.data_dim),
            ]
        )

    def __call__(self, x, t, condition):
        t_emb = self.time_emb(t)
        input = jnp.concatenate([x, t_emb, condition], axis=-1)
        return self.net(input)


##############################################
# deep noise model
##############################################


class PredictorResNet(nn.Module):
    data_dim: int
    condition_dim: int
    width: int = 256
    depth: int = 16  # 1 residual block = 4 layers
    t_dim: int = 16

    def setup(self):
        self.time_emb = nn.Sequential(
            [
                SinusoidalPosEmb(self.t_dim),
                nn.Dense(self.t_dim * 2),
                mish,
                nn.Dense(self.t_dim),
            ]
        )
        self.input_proj = nn.Dense(self.width)
        self.t_proj = nn.Dense(self.width)
        num_blocks = self.depth // 4
        self.blocks = [ResidualBlock(width=self.width, norm_cls=nn.LayerNorm, activation=mish) for _ in range(num_blocks)]
        self.out_proj = nn.Dense(self.data_dim)

    def __call__(self, x, t, condition):
        t_emb = self.time_emb(t)  # shape: (batch, t_dim)
        inputs = jnp.concatenate([x, condition], axis=-1)
        inputs = self.input_proj(inputs)  # shape: (batch, width)
        t_proj = self.t_proj(t_emb)  # shape: (batch, width)
        for block in self.blocks:
            inputs = inputs + t_proj  # shape: (batch, width)
            inputs = block(inputs)
        return self.out_proj(inputs)


##############################################
# conditioned diffusion model
##############################################


class ConditionedDiffusion(nn.Module):
    data_dim: int  # 生成数据的维度
    condition_dim: int  # 条件的维度
    max_data: float  # 数据的最大值
    beta_schedule: str = "vp"
    n_timesteps: int = 100
    loss_type: str = "l2"
    clip_denoised: bool = True
    predict_epsilon: bool = True
    predictor: nn.Module = PredictorMLP

    def setup(self):
        self.model = self.predictor(data_dim=self.data_dim, condition_dim=self.condition_dim)
        if self.loss_type == "l2":
            self.loss_fn = lambda pred, target, weights: jnp.mean(((pred - target) ** 2) * weights)
        elif self.loss_type == "l1":
            self.loss_fn = lambda pred, target, weights: jnp.mean(jnp.abs(pred - target) * weights)
        else:
            raise NotImplementedError(f"loss type {self.loss_type} not implemented")

        if self.beta_schedule == "linear":
            betas = linear_beta_schedule(self.n_timesteps)
        elif self.beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.n_timesteps)
        elif self.beta_schedule == "vp":
            betas = vp_beta_schedule(self.n_timesteps)
        else:
            raise ValueError(f"unknown beta_schedule: {self.beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = jnp.cumprod(alphas, axis=0)
        alphas_cumprod_prev = jnp.concatenate([jnp.ones((1,), dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]], axis=0)
        sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - alphas_cumprod)
        log_one_minus_alphas_cumprod = jnp.log(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = jnp.sqrt(1.0 / alphas_cumprod - 1.0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance_clipped = jnp.log(jnp.clip(posterior_variance, a_min=1e-20))
        posterior_mean_coef1 = betas * jnp.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * jnp.sqrt(alphas) / (1.0 - alphas_cumprod)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod
        self.log_one_minus_alphas_cumprod = log_one_minus_alphas_cumprod
        self.sqrt_recip_alphas_cumprod = sqrt_recip_alphas_cumprod
        self.sqrt_recipm1_alphas_cumprod = sqrt_recipm1_alphas_cumprod
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = posterior_log_variance_clipped
        self.posterior_mean_coef1 = posterior_mean_coef1
        self.posterior_mean_coef2 = posterior_mean_coef2

    def predict_start_from_noise(self, x_t: jnp.ndarray, t: jnp.ndarray, noise: jnp.ndarray) -> jnp.ndarray:
        if self.predict_epsilon:
            return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        else:
            return noise

    def q_posterior(self, x_start: jnp.ndarray, x_t: jnp.ndarray, t: jnp.ndarray):
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x: jnp.ndarray, t: jnp.ndarray, condition: jnp.ndarray):
        noise_pred = self.model(x, t, condition)
        x_recon = self.predict_start_from_noise(x, t, noise_pred)
        if self.clip_denoised:
            x_recon = jnp.clip(x_recon, -self.max_data, self.max_data)
        model_mean, model_var, model_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, model_var, model_log_variance

    def p_sample(self, x: jnp.ndarray, t: jnp.ndarray, condition: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        model_mean, _, model_log_variance = self.p_mean_variance(x, t, condition)
        rng, subkey = jax.random.split(rng)
        noise = jax.random.normal(subkey, shape=x.shape)
        nonzero_mask = (t != 0).astype(x.dtype)
        nonzero_mask = nonzero_mask.reshape((nonzero_mask.shape[0],) + (1,) * (len(x.shape) - 1))
        return model_mean + nonzero_mask * jnp.exp(0.5 * model_log_variance) * noise

    def p_sample_loop(self, condition: jnp.ndarray, shape: tuple, rng: jnp.ndarray, verbose: bool = False, return_diffusion: bool = False) -> Any:
        rng, subkey = jax.random.split(rng)
        x = jax.random.normal(subkey, shape)
        diffusion = [x] if return_diffusion else None
        for i in reversed(range(self.n_timesteps)):
            t = jnp.full((shape[0],), i, dtype=jnp.int32)
            rng, subkey = jax.random.split(rng)
            x = self.p_sample(x, t, condition, subkey)
            if verbose:
                print(f"t = {i}")
            if return_diffusion:
                diffusion.append(x)
        if return_diffusion:
            diffusion = jnp.stack(diffusion, axis=1)
            return x, diffusion
        else:
            return x

    def sample(self, condition: jnp.ndarray, rng: jnp.ndarray, **kwargs) -> jnp.ndarray:
        batch_size = condition.shape[0]
        shape = (batch_size, self.data_dim)
        rng, subkey = jax.random.split(rng)
        data = self.p_sample_loop(condition, shape, subkey, **kwargs)
        return jnp.clip(data, -self.max_data, self.max_data)

    def q_sample(self, x_start: jnp.ndarray, t: jnp.ndarray, rng: jnp.ndarray, noise: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if noise is None:
            noise = jax.random.normal(rng, shape=x_start.shape)
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def p_losses(self, x_start: jnp.ndarray, condition: jnp.ndarray, t: jnp.ndarray, rng: jnp.ndarray, weights: float = 1.0) -> jnp.ndarray:
        rng, subkey = jax.random.split(rng)
        noise = jax.random.normal(subkey, shape=x_start.shape)
        rng, subkey = jax.random.split(rng)
        x_noisy = self.q_sample(x_start, t, subkey, noise)
        x_recon = self.model(x_noisy, t, condition)
        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)
        return loss

    def loss(
        self,
        x_start: jnp.ndarray,  # (batch, data_dim)
        condition: jnp.ndarray,  # (batch, condition_dim)
        rng: jnp.ndarray,
        weights: float = 1.0,
    ) -> jnp.ndarray:
        batch_size = x_start.shape[0]
        rng, subkey = jax.random.split(rng)
        t = jax.random.randint(subkey, shape=(batch_size,), minval=0, maxval=self.n_timesteps)
        return self.p_losses(x_start, condition, t, rng, weights)

    def __call__(self, condition: jnp.ndarray, rng: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self.sample(condition, rng, **kwargs)
