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

# 导入严格验证模块
try:
    from verl.utils.debug.igpo_pipeline_checker import (
        is_strict_check_enabled,
        record_core_algos_rewards,
        record_normalization_stats,
        record_turn_level_results,
        run_all_checks,
    )
    _HAS_STRICT_CHECK = True
except ImportError:
    _HAS_STRICT_CHECK = False
    def is_strict_check_enabled(): return False


def _compute_turn_level_advantage(
    normalized_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    bsz: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Turn-level 折扣累积 + 广播实现。
    
    每个 turn 由 reward 位置定义（非零 reward 标记 turn 的结束）。
    
    计算流程：
    1. 识别每个样本的 turn 边界（基于 reward 位置）
    2. Turn-level 折扣累积：A_i = r_i + gamma * A_{i+1}
    3. 广播：将 A_i 广播到 turn i 的所有 tokens
    
    Args:
        normalized_rewards: 归一化后的 rewards (bsz, seq_len)
        response_mask: 响应掩码 (bsz, seq_len)
        gamma: 折扣因子
        bsz: batch size
        seq_len: 序列长度
        device: 设备
    
    Returns:
        discounted_returns: Turn-level advantage 广播到所有 tokens (bsz, seq_len)
    """
    import os
    debug_turn_level = os.environ.get("DEBUG_TURN_LEVEL_ADV", "0") == "1"
    strict_check = _HAS_STRICT_CHECK and is_strict_check_enabled()
    
    discounted_returns = torch.zeros(bsz, seq_len, device=device, dtype=normalized_rewards.dtype)
    
    # 用于 debug 统计
    total_samples_with_rewards = 0
    total_turns = 0
    
    for sample_idx in range(bsz):
        sample_rewards = normalized_rewards[sample_idx]  # (seq_len,)
        sample_mask = response_mask[sample_idx]  # (seq_len,)
        
        # Step 1: 找到所有 reward 位置（turn 结束位置）
        reward_positions = (sample_rewards != 0).nonzero(as_tuple=True)[0].tolist()
        
        if len(reward_positions) == 0:
            # 没有 reward，跳过
            continue
        
        total_samples_with_rewards += 1
        total_turns += len(reward_positions)
        
        # Step 2: Turn-level 折扣累积（从后向前）
        # turn_data: [(reward_pos, turn_advantage), ...]
        turn_data = []
        next_turn_adv = 0.0
        
        for pos in reversed(reward_positions):
            turn_reward = sample_rewards[pos].item()
            turn_adv = turn_reward + gamma * next_turn_adv
            turn_data.append((pos, turn_adv))
            next_turn_adv = turn_adv
        
        turn_data.reverse()  # 变成从前到后的顺序
        
        # Step 3: 广播到每个 turn 的所有 tokens
        # Turn i 的范围：[prev_reward_pos + 1, current_reward_pos]
        # 第一个 turn 从位置 0 开始
        prev_end = 0
        for i, (reward_pos, adv) in enumerate(turn_data):
            # Turn 范围：[prev_end, reward_pos]
            # 只广播到 response_mask == 1 的位置
            for t in range(prev_end, reward_pos + 1):
                if sample_mask[t] == 1:
                    discounted_returns[sample_idx, t] = adv
            prev_end = reward_pos + 1
        
        # ========== 严格验证：记录 turn-level 结果（只记录 turn_data）==========
        if strict_check:
            record_turn_level_results(sample_idx, turn_data)
        
        # ========== DEBUG: 验证单个样本 ==========
        if debug_turn_level and sample_idx < 3:  # 只打印前 3 个样本
            print(f"\n[Turn-Level DEBUG] === Sample {sample_idx} ===")
            print(f"  Reward positions: {reward_positions}")
            reward_values = [f"{sample_rewards[p].item():.4f}" for p in reward_positions]
            print(f"  Reward values: {reward_values}")
            print(f"  Gamma: {gamma}")
            print(f"  Turn advantages (from turn-level discounting):")
            for i, (pos, adv) in enumerate(turn_data):
                print(f"    Turn {i}: pos={pos}, advantage={adv:.4f}")
            
            # 验证折扣累积公式
            print(f"  Verification of discounting formula A_i = r_i + γ * A_{i+1}:")
            for i in range(len(turn_data) - 1, -1, -1):
                pos, adv = turn_data[i]
                r_i = sample_rewards[pos].item()
                if i == len(turn_data) - 1:
                    expected = r_i
                    print(f"    Turn {i}: A_{i} = r_{i} = {r_i:.4f}, actual={adv:.4f}, match={abs(expected - adv) < 1e-6}")
                else:
                    A_next = turn_data[i + 1][1]
                    expected = r_i + gamma * A_next
                    print(f"    Turn {i}: A_{i} = {r_i:.4f} + {gamma} * {A_next:.4f} = {expected:.4f}, actual={adv:.4f}, match={abs(expected - adv) < 1e-6}")
    
    # ========== DEBUG: 全局验证 ==========
    if debug_turn_level:
        print(f"\n[Turn-Level DEBUG] === Global Statistics ===")
        print(f"  Total samples with rewards: {total_samples_with_rewards}/{bsz}")
        print(f"  Total turns: {total_turns}")
        print(f"  Average turns per sample: {total_turns / max(total_samples_with_rewards, 1):.2f}")
        
        # 验证：检查每个 turn 内的 tokens 是否都有相同的 advantage
        uniform_count = 0
        non_uniform_count = 0
        
        for sample_idx in range(bsz):
            sample_rewards = normalized_rewards[sample_idx]
            sample_returns = discounted_returns[sample_idx]
            sample_mask = response_mask[sample_idx]
            
            reward_positions = (sample_rewards != 0).nonzero(as_tuple=True)[0].tolist()
            if len(reward_positions) == 0:
                continue
            
            prev_end = 0
            sample_is_uniform = True
            for pos in reward_positions:
                # 检查 [prev_end, pos] 范围内的 advantage 是否一致
                turn_values = []
                for t in range(prev_end, pos + 1):
                    if sample_mask[t] == 1:
                        turn_values.append(sample_returns[t].item())
                
                if len(turn_values) > 1:
                    # 检查是否所有值都相等
                    if not all(abs(v - turn_values[0]) < 1e-6 for v in turn_values):
                        sample_is_uniform = False
                        break
                
                prev_end = pos + 1
            
            if sample_is_uniform:
                uniform_count += 1
            else:
                non_uniform_count += 1
        
        print(f"  Samples with uniform turn advantages: {uniform_count}/{total_samples_with_rewards}")
        if non_uniform_count > 0:
            print(f"  ⚠️  WARNING: {non_uniform_count} samples have non-uniform advantages within turns!")
        else:
            print(f"  ✓ All samples have uniform advantages within each turn")
    
    # ========== 严格验证：在循环结束后记录完整的 discounted_returns ==========
    if strict_check:
        record_turn_level_results(-1, [], discounted_returns)
    
    return discounted_returns


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
    info_gain_norm_mode: str = "joint",
    curriculum_f1_weight: float = 1.0,
    curriculum_ig_weight: float = 1.0,
):
    """
    为 GRPO 计算优势函数 (Advantage)，使用 Turn-level 累积 + 广播。
    
    计算流程：
    1. 归一化 rewards（info_gain 和 f1）
    2. Turn-level 折扣累积：A_i = r_i + gamma * A_{i+1}
    3. 将每个 turn 的 advantage 广播到该 turn 的所有 tokens
    
    Args:
        token_level_rewards: (bs, response_length) 每个词元的即时奖励
        response_mask: (bs, response_length) 响应序列掩码
        index: 用于将样本分组的提示索引数组
        epsilon: 防止除零的小常数
        norm_adv_by_std_in_grpo: 是否除以标准差
        gamma: 折扣因子，默认 1.0
        info_gain_norm_mode: "joint" 或 "separate"
        curriculum_f1_weight: F1 奖励的 Curriculum 权重，默认 1.0
        curriculum_ig_weight: InfoGain 奖励的 Curriculum 权重，默认 1.0

    Returns:
        advantages, returns: 均为 (bs, response_length)
    """
    import os
    debug_pipeline = os.environ.get("DEBUG_IGPO_PIPELINE", "0") == "1"
    strict_check = _HAS_STRICT_CHECK and is_strict_check_enabled()
    
    bsz, seq_len = token_level_rewards.shape
    device = token_level_rewards.device
    
    # 保存原始 rewards 用于验证
    original_rewards = token_level_rewards.clone() if strict_check else None

    # ========== 严格验证：记录接收到的 rewards ==========
    if strict_check:
        record_core_algos_rewards(token_level_rewards, response_mask)

    # ========== DEBUG: 验证点 3 - Reward 传输正确性 ==========
    if debug_pipeline:
        print(f"\n[IGPO Pipeline Check 3] === core_algos.py: Reward Reception ===")
        print(f"  Input shape: ({bsz}, {seq_len})")
        
        # 统计接收到的 rewards
        nonzero_mask = token_level_rewards != 0
        total_nonzero = nonzero_mask.sum().item()
        samples_with_rewards = (nonzero_mask.sum(dim=1) > 0).sum().item()
        
        print(f"  Samples with rewards: {samples_with_rewards}/{bsz}")
        print(f"  Total non-zero rewards: {total_nonzero}")
        print(f"  Average rewards per sample: {total_nonzero / max(samples_with_rewards, 1):.2f}")
        
        # 打印前 3 个样本的详细信息
        for i in range(min(3, bsz)):
            sample_rewards = token_level_rewards[i]
            reward_positions = (sample_rewards != 0).nonzero(as_tuple=True)[0].tolist()
            reward_values = [f"{sample_rewards[p].item():.6f}" for p in reward_positions]
            print(f"\n  Sample {i}:")
            print(f"    Reward positions: {reward_positions}")
            print(f"    Reward values: {reward_values}")
            
            # 分析 reward 类型（info_gain vs f1）
            if len(reward_positions) > 0:
                last_pos = reward_positions[-1]
                valid_len = response_mask[i].sum().item()
                is_last_f1 = (last_pos == valid_len - 1)
                print(f"    Valid length: {int(valid_len)}, Last reward at: {last_pos}")
                print(f"    Last reward is F1: {is_last_f1} {'✓' if is_last_f1 else '⚠️'}")

    # ========== Step 1: 构建掩码 ==========
    with torch.no_grad():
        valid_lengths = response_mask.sum(dim=1).long()
        last_valid_pos = torch.clamp(valid_lengths - 1, min=0)
        
        position_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        f1_mask = (position_indices == last_valid_pos.unsqueeze(1)) & (response_mask == 1)
        ig_mask = (response_mask == 1) & (~f1_mask) & (token_level_rewards != 0)
    
    # ========== Step 1.5: 应用 Curriculum 权重 ==========
    if curriculum_f1_weight != 1.0 or curriculum_ig_weight != 1.0:
        weighted_rewards = token_level_rewards.clone()
        weighted_rewards = torch.where(f1_mask, token_level_rewards * curriculum_f1_weight, weighted_rewards)
        weighted_rewards = torch.where(ig_mask, token_level_rewards * curriculum_ig_weight, weighted_rewards)
        token_level_rewards = weighted_rewards

    # ========== Step 2: 构建 Group 映射（向量化） ==========
    # 将 index 转换为连续的 group_id (0, 1, 2, ...)
    unique_indices, inverse_indices = np.unique(index, return_inverse=True)
    group_ids = torch.tensor(inverse_indices, device=device, dtype=torch.long)  # (bsz,)
    num_groups = len(unique_indices)
    
    # 扩展 group_ids 到 (bsz, seq_len)
    group_ids_expanded = group_ids.unsqueeze(1).expand(-1, seq_len)

    # ========== Step 3: 向量化计算组内统计量 ==========
    def compute_group_stats(mask):
        """计算每个 group 在 mask 位置的 mean 和 std"""
        flat_mask = mask.view(-1)
        flat_rewards = token_level_rewards.view(-1)
        flat_group_ids = group_ids_expanded.reshape(-1)
        
        # 只选取有效位置
        valid_idx = flat_mask.nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            return torch.zeros(num_groups, device=device), torch.ones(num_groups, device=device)
        
        valid_rewards = flat_rewards[valid_idx]
        valid_groups = flat_group_ids[valid_idx]
        
        # 计算 sum 和 count
        group_sum = torch.zeros(num_groups, device=device).scatter_add_(0, valid_groups, valid_rewards)
        group_count = torch.zeros(num_groups, device=device).scatter_add_(0, valid_groups, torch.ones_like(valid_rewards))
        
        # Mean
        group_mean = group_sum / group_count.clamp(min=1.0)
        
        # Std: 使用 E[(x - mean)^2] 公式
        expanded_mean = group_mean[valid_groups]
        sq_diff = (valid_rewards - expanded_mean) ** 2
        group_sq_sum = torch.zeros(num_groups, device=device).scatter_add_(0, valid_groups, sq_diff)
        group_var = group_sq_sum / group_count.clamp(min=1.0)
        group_std = torch.sqrt(group_var + 1e-8)
        
        # count <= 1 时，std 设为 1.0
        group_std = torch.where(group_count <= 1, torch.ones_like(group_std), group_std)
        
        return group_mean, group_std

    # ========== Step 4: 向量化归一化 ==========
    normalized_rewards = torch.zeros_like(token_level_rewards)

    if info_gain_norm_mode == "separate":
        # F1 部分
        f1_mean, f1_std = compute_group_stats(f1_mask)
        f1_mean_map = f1_mean[group_ids_expanded]
        f1_std_map = f1_std[group_ids_expanded]
        
        norm_f1 = (token_level_rewards - f1_mean_map)
        if norm_adv_by_std_in_grpo:
            norm_f1 = norm_f1 / (f1_std_map + epsilon)
        normalized_rewards = torch.where(f1_mask, norm_f1, normalized_rewards)
        
        # InfoGain 部分
        ig_mean, ig_std = compute_group_stats(ig_mask)
        ig_mean_map = ig_mean[group_ids_expanded]
        ig_std_map = ig_std[group_ids_expanded]
        
        norm_ig = (token_level_rewards - ig_mean_map)
        if norm_adv_by_std_in_grpo:
            norm_ig = norm_ig / (ig_std_map + epsilon)
        normalized_rewards = torch.where(ig_mask, norm_ig, normalized_rewards)
    
    else:  # joint
        joint_mask = f1_mask | ig_mask
        g_mean, g_std = compute_group_stats(joint_mask)
        mean_map = g_mean[group_ids_expanded]
        std_map = g_std[group_ids_expanded]
        
        norm_val = (token_level_rewards - mean_map)
        if norm_adv_by_std_in_grpo:
            norm_val = norm_val / (std_map + epsilon)
        normalized_rewards = torch.where(joint_mask, norm_val, normalized_rewards)

    # ========== DEBUG: 验证点 4 - 标准化正确性 ==========
    if debug_pipeline:
        print(f"\n[IGPO Pipeline Check 4] === core_algos.py: Normalization ===")
        print(f"  Normalization mode: {info_gain_norm_mode}")
        print(f"  Norm by std: {norm_adv_by_std_in_grpo}")
        print(f"  Number of groups: {num_groups}")
        
        # 统计归一化前后的 rewards
        f1_count = f1_mask.sum().item()
        ig_count = ig_mask.sum().item()
        print(f"  F1 rewards count: {f1_count}")
        print(f"  InfoGain rewards count: {ig_count}")
        
        # 验证归一化结果
        if info_gain_norm_mode == "separate":
            # 分别检查 F1 和 InfoGain
            f1_values = token_level_rewards[f1_mask]
            ig_values = token_level_rewards[ig_mask]
            f1_norm_values = normalized_rewards[f1_mask]
            ig_norm_values = normalized_rewards[ig_mask]
            
            if len(f1_values) > 0:
                print(f"\n  F1 Rewards:")
                print(f"    Before norm - mean: {f1_values.mean().item():.4f}, std: {f1_values.std().item():.4f}")
                print(f"    After norm  - mean: {f1_norm_values.mean().item():.4f}, std: {f1_norm_values.std().item():.4f}")
            
            if len(ig_values) > 0:
                print(f"\n  InfoGain Rewards:")
                print(f"    Before norm - mean: {ig_values.mean().item():.4f}, std: {ig_values.std().item():.4f}")
                print(f"    After norm  - mean: {ig_norm_values.mean().item():.4f}, std: {ig_norm_values.std().item():.4f}")
        else:
            # Joint
            joint_values = token_level_rewards[joint_mask]
            joint_norm_values = normalized_rewards[joint_mask]
            
            if len(joint_values) > 0:
                print(f"\n  Joint (F1 + InfoGain) Rewards:")
                print(f"    Before norm - mean: {joint_values.mean().item():.4f}, std: {joint_values.std().item():.4f}")
                print(f"    After norm  - mean: {joint_norm_values.mean().item():.4f}, std: {joint_norm_values.std().item():.4f}")
        
        # 验证归一化后非零位置是否正确保留
        nonzero_before = (token_level_rewards != 0).sum().item()
        nonzero_after = (normalized_rewards != 0).sum().item()
        print(f"\n  Non-zero count before norm: {nonzero_before}")
        print(f"  Non-zero count after norm: {nonzero_after}")
        print(f"  Positions preserved: {nonzero_before == nonzero_after} {'✓' if nonzero_before == nonzero_after else '⚠️'}")

    # ========== 严格验证：记录归一化统计 ==========
    if strict_check and original_rewards is not None:
        record_normalization_stats(
            before_rewards=original_rewards,
            after_rewards=normalized_rewards,
            f1_mask=f1_mask,
            ig_mask=ig_mask,
            mode=info_gain_norm_mode,
            group_ids=group_ids,
        )

    # ========== Step 5: Turn-level 折扣累积 + 广播 ==========
    # 每个 turn 的 advantage 通过 turn-level 折扣累积计算
    # 然后广播到该 turn 的所有 tokens
    discounted_returns = _compute_turn_level_advantage(
        normalized_rewards=normalized_rewards,
        response_mask=response_mask,
        gamma=gamma,
        bsz=bsz,
        seq_len=seq_len,
        device=device,
    )

    # ========== 严格验证：运行所有检查 ==========
    if strict_check:
        run_all_checks(
            gamma=gamma,
            norm_mode=info_gain_norm_mode,
            norm_by_std=norm_adv_by_std_in_grpo,
            response_mask=response_mask,
            normalized_rewards=normalized_rewards,
        )

    return discounted_returns, discounted_returns




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
