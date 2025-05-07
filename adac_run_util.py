import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxrl_m.typing import *
from dataclasses import dataclass
import heapq,copy


def get_name_suffix(use_time=True, **kwargs):
    parts = [f"|{k}={v}" for k, v in kwargs.items()]
    if use_time:
        import time

        parts.append(time.strftime("|%Y%m%d-%H%M%S"))
    return "".join(parts)


##############################################
# reward tuning
##############################################


def get_iql_normalization(dataset):
    returns = []
    ret = 0
    for r, term in zip(dataset["rewards"], dataset["dones_float"]):
        ret += r
        if term:
            returns.append(ret)
            ret = 0
    return (max(returns) - min(returns)) / 1000


def get_tuned_dataset(dataset, reward_tune):
    if reward_tune == "no":
        pass
    elif reward_tune == "iql_locomotion":
        normalizing_factor = get_iql_normalization(dataset)
        dataset = dataset.copy({"rewards": dataset["rewards"] / normalizing_factor})
    elif reward_tune == "normalize":
        mean = dataset["rewards"].mean()
        std = dataset["rewards"].std()
        dataset = dataset.copy({"rewards": (dataset["rewards"] - mean) / std})
    elif reward_tune == "cql_antmaze":
        dataset = dataset.copy({"rewards": (dataset["rewards"] - 0.5) * 4.0})
    elif reward_tune == "iql_antmaze":
        dataset = dataset.copy({"rewards": dataset["rewards"] - 1.0})
    elif reward_tune == "antmaze":
        dataset = dataset.copy({"rewards": (dataset["rewards"] - 0.25) * 2.0})
    return dataset


##############################################
# topk ckpt
##############################################

import heapq
import copy
from dataclasses import dataclass

@dataclass
class Checkpoint:
    bc_loss: float
    eval_score: float
    step: int

class TopKHeap:
    def __init__(self, k, ms_type: str = "offline"):
        self.k = k
        self.ms_type = ms_type
        # 堆中存储 (sort_key, step, checkpoint)
        self.heap = []

    def add(self, bc_loss: float, eval_score: float, step: int):
        checkpoint = Checkpoint(bc_loss, eval_score, step)
        if self.ms_type == "offline":
            # 离线模式：目标是 bc_loss 越小越好，
            # 用负的 bc_loss 作为排序键，这样堆顶就是 bc_loss 最大（最差）的元素
            key = -bc_loss
        elif self.ms_type == "online":
            # 在线模式：目标是 eval_score 越大越好，
            # 直接用 eval_score 作为排序键，堆顶是最小 eval_score（最差）的元素
            key = eval_score
        else:
            raise ValueError("ms_type must be either 'offline' or 'online'.")

        item = (key, -step, checkpoint)  # 在 key 相同时，使用 step 作为 tie-breaker, 且 step 越大越好
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, item)
        else:
            if key > self.heap[0][0]:
                heapq.heapreplace(self.heap, item)
                
    def get_all(self):
        # 返回所有 Checkpoint 对象
        sorted_heap = sorted(self.heap, reverse=True)
        return [item[2] for item in sorted_heap]
    
    

def safe_convert(val):
    """
    尝试将可能为 JAX 或 NumPy 类型的值转换为 Python 内置的 float 类型。
    如果无法转换，则直接返回原始值。
    """
    if hasattr(val, "item"):
        try:
            return val.item()
        except Exception:
            return val
    return val


def convert_checkpoint_dict(d):
    """
    将 checkpoint 字典中每个字段转换为 Python 内置类型，确保 YAML 可读性。
    """
    return {k: safe_convert(v) for k, v in d.items()}

    

