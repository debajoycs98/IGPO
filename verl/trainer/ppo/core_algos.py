# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from collections import defaultdict

import numpy as np
import torch

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    gamma: float = 1.0,
):
    """
    为 GRPO 计算优势函数 (Advantage)。
    
    计算顺序：先进行组内全局归一化，再计算折扣累积回报。
    
    步骤：
    1. 对每个 group 内的所有 rewards 值计算全局均值和标准差
    2. 归一化：normalized_rewards = (rewards - mean) / (std + epsilon)
    3. 使用归一化后的 rewards 计算折扣累积回报：R_t = r_t + gamma * R_{t+1}

    Args:
        token_level_rewards: `(torch.Tensor)`
            形状为 (bs, response_length) 的每个词元的即时奖励。
        response_mask: `(torch.Tensor)`
            形状为 (bs, response_length) 的响应序列掩码。
        index: `(np.ndarray)`
            用于将样本分组的提示 (prompt) 索引数组。
        epsilon: `(float)`
            用于在归一化时防止除以零的小常数。
        norm_adv_by_std_in_grpo: (bool)
            是否通过标准差来缩放优势。
            若为 True，则除以标准差（原始 GRPO 的做法）。
            若为 False，则不进行缩放（类似 Dr.GRPO 的做法）。
        gamma: (float)
            折扣因子，用于计算折扣累积回报 R_t = r_t + gamma * R_{t+1}。
            默认值为 1.0（无折扣）。

    Returns:
        advantages: `(torch.Tensor)` 形状为 (bs, response_length)
            经过归一化和折扣累积后的优势值。
        returns: `(torch.Tensor)` 形状为 (bs, response_length)
            折扣累积回报。
    """
    bsz, seq_len = token_level_rewards.shape
    device = token_level_rewards.device

    # ========== Step 1: 组内全局归一化 ==========
    # 按 index 将所有样本的 token_level_rewards 分组，
    # 对整个 group 的所有 rewards 值计算一个全局均值和标准差
    
    id2indices = defaultdict(list)  # 记录每个组包含哪些样本索引
    id2mean = {}  # 每个组的全局均值 (scalar)
    id2std = {}   # 每个组的全局标准差 (scalar)

    with torch.no_grad():
        # 首先，按 index 将样本索引分组
        for i in range(bsz):
            id2indices[index[i]].append(i)
        
        # 为每个组计算全局均值和标准差
        for idx, sample_indices in id2indices.items():
            # 收集该组内所有样本的所有 rewards 值
            group_rewards = []
            for i in sample_indices:
                # 只收集有效位置（mask=1）的 rewards
                valid_rewards = token_level_rewards[i][response_mask[i] == 1]
                group_rewards.append(valid_rewards)
            
            # 将所有 rewards 拼接成一个一维张量
            all_rewards = torch.cat(group_rewards, dim=0)  # (total_valid_tokens,)
            
            if all_rewards.numel() <= 1:
                # 如果只有一个有效值，标准差设为1
                id2mean[idx] = all_rewards.mean() if all_rewards.numel() == 1 else torch.tensor(0.0, device=device)
                id2std[idx] = torch.tensor(1.0, device=device)
            else:
                # 计算全局均值和标准差
                id2mean[idx] = all_rewards.mean()  # scalar
                id2std[idx] = all_rewards.std()    # scalar

    # 对 token_level_rewards 进行全局归一化
    normalized_rewards = torch.zeros_like(token_level_rewards)
    for i in range(bsz):
        idx = index[i]
        mean_val = id2mean[idx]  # scalar
        std_val = id2std[idx]    # scalar
        
        # 归一化：(rewards - mean) / (std + epsilon)
        normalized_rewards[i] = token_level_rewards[i] - mean_val
        if norm_adv_by_std_in_grpo:
            normalized_rewards[i] = normalized_rewards[i] / (std_val + epsilon)

    # ========== Step 2: 计算折扣累积回报 ==========
    # 使用归一化后的 rewards 计算 R_t = r_t + gamma * R_{t+1}
    
    discounted_returns = torch.zeros_like(normalized_rewards)
    next_return = torch.zeros(bsz, device=device)
    
    for t in reversed(range(seq_len)):
        current_reward = normalized_rewards[:, t]
        current_mask = response_mask[:, t]
        current_return = current_reward + gamma * next_return
        discounted_returns[:, t] = current_return
        next_return = current_return * current_mask

    # ========== Step 3: 应用掩码 ==========
    advantages = discounted_returns * response_mask
    final_returns = discounted_returns * response_mask

    return advantages, final_returns




# # NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
# def compute_multi_turn_grpo_outcome_advantage(
#     token_level_rewards: torch.Tensor,
#     response_mask: torch.Tensor,
#     index: np.ndarray,
#     epsilon: float = 1e-6,
#     norm_adv_by_std_in_grpo: str = True,
# ):
#     """
#     Compute advantage for GRPO, operating only on Outcome reward
#     (with only one scalar reward for each response).
#     Args:
#         token_level_rewards: `(torch.Tensor)`
#             shape: (bs, response_length)
#         response_mask: `(torch.Tensor)`
#             shape: (bs, response_length)
#         norm_adv_by_std_in_grpo: (bool)
#             whether to scale the GRPO advantage.
#             If True, the advantage is scaled by the std, as in the original GRPO.
#             If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

#     Returns:
#         advantages: `(torch.Tensor)`
#             shape: (bs, response_length)
#         Returns: `(torch.Tensor)`
#             shape: (bs, response_length)
#     """


#     """
#     为 GRPO 计算带有折扣奖励的优势函数 (Advantage)。
    
#     每个词元的回报 (return) R_t 通过从序列末尾向前递归计算：
#     R_t = r_t + gamma * R_{t+1}

#     Args:
#         token_level_rewards: `(torch.Tensor)`
#             形状为 (bs, response_length) 的每个词元的即时奖励。
#         response_mask: `(torch.Tensor)`
#             形状为 (bs, response_length) 的响应序列掩码。
#         index: `(np.ndarray)`
#             用于将样本分组的提示 (prompt) 索引数组。
#         gamma: `(float)`
#             未来奖励的折扣因子
#         norm_adv_by_std_in_grpo: (bool)
#             是否通过标准差来缩放 GRPO 优势。
#             若为 True，则优势会被标准差缩放（原始 GRPO 的做法）。
#             若为 False，则不进行缩放（类似 Dr.GRPO 的做法 https://arxiv.org/abs/2503.20783）。

#     Returns:
#         advantages: `(torch.Tensor)` 形状为 (bs, response_length)
#             经过基线扣除和归一化后的每个词元的优势值。
#         returns: `(torch.Tensor)` 形状为 (bs, response_length)
#             每个词元的折扣累积奖励 (G_t)。
#     """
#     bsz, seq_len = token_level_rewards.shape
#     device = token_level_rewards.device

#     # 1. 计算每个词元的折扣累积奖励 (returns)
#     #    采用从序列末尾到开头的反向递归方式。
#     discounted_returns = torch.zeros_like(token_level_rewards)
#     # 初始化下一个时间步的回报为0
#     next_return = torch.zeros(bsz, device=device)
    
#     gamma = 0.88
    
#     for t in reversed(range(seq_len)):
#         # 获取当前时间步的奖励和掩码
#         current_reward = token_level_rewards[:, t]
#         current_mask = response_mask[:, t]
        
#         # 计算当前时间步的回报 R_t = r_t + gamma * R_{t+1}
#         current_return = current_reward + gamma * next_return
        
#         # 存储计算出的回报
#         discounted_returns[:, t] = current_return
        
#         # 更新下一个时间步的回报，并应用掩码。
#         # 如果当前词元是填充符 (mask=0)，其回报不会影响到前一个词元。
#         next_return = current_return * current_mask

#     # 2. 计算用于归一化的基线 (mean) 和标准差 (std)
#     #    我们使用每个序列的总回报 R_0 (即第一个时间步的回报) 作为其“得分”来进行比较。
#     scores = discounted_returns[:, 0].clone()

#     id2score = defaultdict(list)
#     id2mean = {}
#     id2std = {}

#     with torch.no_grad():
#         for i in range(bsz):
#             id2score[index[i]].append(scores[i])
        
#         for idx in id2score:
#             group_scores = torch.tensor(id2score[idx], device=device)
#             if len(group_scores) <= 1:
#                 # 如果组内只有一个样本，其优势为0。
#                 # 将均值设为样本自身的值，标准差设为1以避免除零。
#                 id2mean[idx] = group_scores.mean() if len(group_scores) == 1 else torch.tensor(0.0, device=device)
#                 id2std[idx] = torch.tensor(1.0, device=device)
#             else:
#                 id2mean[idx] = group_scores.mean()
#                 id2std[idx] = group_scores.std()

#     # 3. 通过从回报中减去基线来计算优势
#     advantages = torch.zeros_like(discounted_returns)
#     for i in range(bsz):
#         idx = index[i]
#         mean_baseline = id2mean[idx]
#         std_dev = id2std[idx]
        
#         # 优势 = 回报 - 基线
#         # A_t = R_t - E[R_0]
#         # 我们使用同一组提示下所有样本总回报 R_0 的均值作为基线。
#         advantages[i] = discounted_returns[i] - mean_baseline
        
#         if norm_adv_by_std_in_grpo:
#             # 根据同一组总回报的标准差进行归一化
#             advantages[i] /= (std_dev + epsilon)

#     # 4. 对最终结果应用掩码，确保填充部分的值为0
#     advantages = advantages * response_mask
#     final_returns = discounted_returns * response_mask

#     return advantages, final_returns

def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask)

    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode="token-mean",
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        clip_ratio_c: (float) default: 3.0
            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
        pg_clipfrac_lower: (float)
            the fraction of policy gradient loss being clipped when the advantage is negative
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=response_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, response_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), response_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == "low_var_kl":
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
