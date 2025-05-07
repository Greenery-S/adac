import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxrl_m.typing import *
from jax.scipy.stats.norm import logpdf

##############################################
# loss and inference utils
##############################################


def sample_from_norm(
    means: jnp.ndarray,
    log_stds: jnp.ndarray,
    key: PRNGKey,
    temperature: float = 1.0,
) -> jnp.ndarray:
    scaled_stds = jnp.exp(log_stds) * temperature
    samples = means + scaled_stds * temperature * jax.random.normal(
        key,
        shape=means.shape,
    )
    return samples


def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


# weighted mse loss
def msew_loss(pred_mean: jnp.ndarray, pred_logstd: jnp.ndarray, gt: jnp.ndarray) -> jnp.ndarray:
    pred_logvar = 2 * pred_logstd
    weighted_mse = jnp.square(pred_mean - gt) * jnp.exp(-pred_logvar)
    return jnp.mean(jnp.mean(weighted_mse, axis=-1))


def var_loss(pred_logstd: jnp.ndarray) -> jnp.ndarray:
    pred_logvar = 2 * pred_logstd
    return jnp.mean(jnp.mean(pred_logvar, axis=-1))


def nll_loss(pred_means: jnp.ndarray, pred_logstds, gt: jnp.ndarray) -> jnp.ndarray:
    return -logpdf(gt, pred_means, jnp.exp(2 * pred_logstds)).mean()


def l1_loss(pred: jnp.ndarray, gt: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(pred - gt))


def l2_loss(pred: jnp.ndarray, gt: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.square(pred - gt))


##############################################
# advantage utils
##############################################


def norm_clip(x: jnp.ndarray, max_norm: float, ord: int = 2, eps: float = 1e-6) -> jnp.ndarray:
    norm = jnp.linalg.norm(x, ord=ord)
    clip_coef = jnp.minimum(1.0, max_norm / (norm + eps))
    return x * clip_coef, norm


def tanh_scale(x: jnp.ndarray, min_scale: float, max_scale: float) -> jnp.ndarray:
    return jnp.where(x < 0, jnp.tanh(x/min_scale) * min_scale, jnp.tanh(x/max_scale) * max_scale)


def drop_extreme(x: jnp.ndarray, min_val: float = -jnp.inf, max_val: float = jnp.inf) -> jnp.ndarray:
    return jnp.where(x < min_val, min_val, jnp.where(x > max_val, max_val, 0))


def soft_clip(x: jnp.ndarray, min_val: Optional[float] = None, max_val: Optional[float] = None, beta: float = 1.5) -> jnp.ndarray:
    _softplus = lambda x, beta: jax.nn.softplus(beta * x) / beta
    if max_val is not None:
        x = max_val - _softplus(max_val - x, beta)
    if min_val is not None:
        x = min_val + _softplus(x - min_val, beta)
    return x


def main():
    nums = jnp.array([1, 2, 3, 4, 5, -999])
    print(norm_clip(nums, 3))


if __name__ == "__main__":
    main()
