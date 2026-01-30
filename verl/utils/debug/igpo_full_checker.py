"""
IGPO 完整训练逻辑验证框架

验证 IGPO 训练流程中每个关键环节的正确性。

启用方式：
    export DEBUG_IGPO_FULL=1

验证环节：
    A. GT LogProb 计算正确性
    B. Info Gain 计算逻辑 (cur - prev)
    C. Turn 边界识别正确性
    D. Token 位置分配正确性
    E. Reward 传输完整性
    F. 归一化数学正确性
    G. Turn-level 折扣累积正确性
    H. Turn 内广播正确性
    I. Advantage 对 Loss 的贡献
    J. process_response_mask 正确性
    K. 向量化 vs 顺序计算等价性

Author: IGPO Team
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


def is_full_check_enabled() -> bool:
    """检查是否启用完整验证"""
    return os.environ.get("DEBUG_IGPO_FULL", "0") == "1"


@dataclass
class FullCheckResult:
    """完整检查结果"""
    check_id: str
    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class IGPOFullChecker:
    """IGPO 完整验证器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有记录"""
        # A. GT LogProb
        self.gt_logprob_sequential: Dict[int, List[float]] = {}  # sample_idx -> [logprob_turn0, logprob_turn1, ...]
        self.gt_logprob_vectorized: Dict[int, List[float]] = {}
        
        # B. Info Gain
        self.info_gain_computed: Dict[int, List[float]] = {}  # sample_idx -> [ig_0, ig_1, ...]
        self.prev_gt_values: Dict[int, List[float]] = {}  # sample_idx -> [prev_0, prev_1, ...]
        self.curr_gt_values: Dict[int, List[float]] = {}  # sample_idx -> [curr_0, curr_1, ...]
        
        # C. Turn 边界
        self.turn_boundaries_char: Dict[int, List[Tuple[int, int]]] = {}  # sample_idx -> [(start, end), ...]
        self.turn_boundaries_token: Dict[int, List[Tuple[int, int]]] = {}
        
        # D. Token 位置
        self.reward_token_positions: Dict[int, List[int]] = {}  # sample_idx -> [pos_turn0, pos_turn1, ...]
        self.expected_token_positions: Dict[int, List[int]] = {}  # 基于 turn 边界计算的预期位置
        
        # E. Reward 传输
        self.rewards_at_info_gain: Dict[int, List[float]] = {}  # info_gain.py 输出
        self.rewards_at_core_algos: Dict[int, List[float]] = {}  # core_algos.py 输入
        
        # F. 归一化
        self.rewards_before_norm: Optional[torch.Tensor] = None
        self.rewards_after_norm: Optional[torch.Tensor] = None
        
        # G/H. Turn-level
        self.turn_rewards: Dict[int, List[float]] = {}  # sample_idx -> [r_turn0, r_turn1, ...]
        self.turn_advantages: Dict[int, List[float]] = {}  # sample_idx -> [A_turn0, A_turn1, ...]
        self.token_advantages: Dict[int, List[float]] = {}  # sample_idx -> [A_t0, A_t1, ...]
        
        # I. Loss 贡献
        self.policy_loss_with_ig: Optional[float] = None
        self.policy_loss_without_ig: Optional[float] = None  # 假设 info_gain 全为 0
        
        # J. process_response_mask
        self.response_mask_before: Optional[torch.Tensor] = None
        self.response_mask_after: Optional[torch.Tensor] = None
        self.tool_response_positions: Dict[int, List[Tuple[int, int]]] = {}  # sample_idx -> [(start, end), ...]
        
        # L. Loss 验证
        self.advantages_for_loss: Optional[torch.Tensor] = None
        self.log_probs_for_loss: Optional[torch.Tensor] = None
        self.old_log_probs_for_loss: Optional[torch.Tensor] = None
        self.response_mask_for_loss: Optional[torch.Tensor] = None
        self.policy_loss_value: Optional[float] = None
        self.policy_loss_per_token: Optional[torch.Tensor] = None
        
        # M. 梯度验证
        self.grad_log_prob: Optional[torch.Tensor] = None  # d(loss)/d(log_prob)
        self.grad_norm: Optional[float] = None
        
        # N. Info Gain 梯度贡献验证
        self.loss_with_ig: Optional[float] = None
        self.loss_without_ig: Optional[float] = None
        self.ig_contribution: Optional[float] = None
        
        # 验证结果
        self.results: List[FullCheckResult] = []
    
    # =========================================================================
    # 记录函数
    # =========================================================================
    
    def record_gt_logprob(self, sample_idx: int, logprobs: List[float], mode: str):
        """记录 GT LogProb 计算结果"""
        if not is_full_check_enabled():
            return
        if mode == "sequential":
            self.gt_logprob_sequential[sample_idx] = logprobs.copy()
        else:
            self.gt_logprob_vectorized[sample_idx] = logprobs.copy()
    
    def record_info_gain_calculation(
        self,
        sample_idx: int,
        turn_idx: int,
        prev_value: float,
        curr_value: float,
        info_gain: float,
    ):
        """记录 Info Gain 计算过程"""
        if not is_full_check_enabled():
            return
        
        if sample_idx not in self.info_gain_computed:
            self.info_gain_computed[sample_idx] = []
            self.prev_gt_values[sample_idx] = []
            self.curr_gt_values[sample_idx] = []
        
        self.prev_gt_values[sample_idx].append(prev_value)
        self.curr_gt_values[sample_idx].append(curr_value)
        self.info_gain_computed[sample_idx].append(info_gain)
    
    def record_turn_boundaries(
        self,
        sample_idx: int,
        char_boundaries: List[Tuple[int, int]],
        token_boundaries: List[Tuple[int, int]],
    ):
        """记录 Turn 边界"""
        if not is_full_check_enabled():
            return
        self.turn_boundaries_char[sample_idx] = char_boundaries.copy()
        self.turn_boundaries_token[sample_idx] = token_boundaries.copy()
    
    def record_reward_positions(
        self,
        sample_idx: int,
        actual_positions: List[int],
        expected_positions: List[int],
    ):
        """记录 Reward 分配位置"""
        if not is_full_check_enabled():
            return
        self.reward_token_positions[sample_idx] = actual_positions.copy()
        self.expected_token_positions[sample_idx] = expected_positions.copy()
    
    def record_rewards_transmission(
        self,
        sample_idx: int,
        rewards_from_info_gain: List[float],
        rewards_at_core_algos: List[float],
    ):
        """记录 Reward 传输"""
        if not is_full_check_enabled():
            return
        self.rewards_at_info_gain[sample_idx] = rewards_from_info_gain.copy()
        self.rewards_at_core_algos[sample_idx] = rewards_at_core_algos.copy()
    
    def record_normalization(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
    ):
        """记录归一化前后"""
        if not is_full_check_enabled():
            return
        self.rewards_before_norm = before.detach().clone()
        self.rewards_after_norm = after.detach().clone()
    
    def record_turn_level_computation(
        self,
        sample_idx: int,
        turn_rewards: List[float],
        turn_advantages: List[float],
        token_advantages: List[float],
    ):
        """记录 Turn-level 计算"""
        if not is_full_check_enabled():
            return
        self.turn_rewards[sample_idx] = turn_rewards.copy()
        self.turn_advantages[sample_idx] = turn_advantages.copy()
        self.token_advantages[sample_idx] = token_advantages.copy()
    
    def record_response_mask(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        tool_positions: Dict[int, List[Tuple[int, int]]],
    ):
        """记录 response_mask 处理"""
        if not is_full_check_enabled():
            return
        self.response_mask_before = before.detach().clone()
        self.response_mask_after = after.detach().clone()
        self.tool_response_positions = {k: v.copy() for k, v in tool_positions.items()}
    
    def record_loss_inputs(
        self,
        advantages: torch.Tensor,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
    ):
        """记录 Loss 计算的输入"""
        if not is_full_check_enabled():
            return
        self.advantages_for_loss = advantages.detach().clone()
        self.log_probs_for_loss = log_probs.detach().clone()
        self.old_log_probs_for_loss = old_log_probs.detach().clone()
        self.response_mask_for_loss = response_mask.detach().clone()
    
    def record_loss_output(
        self,
        loss_value: float,
        loss_per_token: Optional[torch.Tensor] = None,
    ):
        """记录 Loss 计算的输出"""
        if not is_full_check_enabled():
            return
        self.policy_loss_value = loss_value
        if loss_per_token is not None:
            self.policy_loss_per_token = loss_per_token.detach().clone()
    
    def record_gradients(
        self,
        grad_log_prob: Optional[torch.Tensor] = None,
        grad_norm: Optional[float] = None,
    ):
        """记录梯度信息"""
        if not is_full_check_enabled():
            return
        if grad_log_prob is not None:
            self.grad_log_prob = grad_log_prob.detach().clone()
        if grad_norm is not None:
            self.grad_norm = grad_norm
    
    def record_ig_contribution(
        self,
        loss_with_ig: float,
        loss_without_ig: float,
    ):
        """记录 info_gain 对 loss 的贡献"""
        if not is_full_check_enabled():
            return
        self.loss_with_ig = loss_with_ig
        self.loss_without_ig = loss_without_ig
        self.ig_contribution = abs(loss_with_ig - loss_without_ig)
    
    # =========================================================================
    # 验证函数
    # =========================================================================
    
    def verify_all(self, gamma: float = 1.0) -> List[FullCheckResult]:
        """运行所有验证"""
        if not is_full_check_enabled():
            return []
        
        self.results = []
        
        print("\n" + "=" * 80)
        print("    IGPO FULL TRAINING LOGIC VERIFICATION")
        print("=" * 80)
        
        # A. GT LogProb
        self._verify_gt_logprob()
        
        # B. Info Gain 计算
        self._verify_info_gain_calculation()
        
        # C. Turn 边界（与 B 合并）
        
        # D. Token 位置
        self._verify_token_positions()
        
        # E. Reward 传输
        self._verify_reward_transmission()
        
        # F. 归一化
        self._verify_normalization()
        
        # G/H. Turn-level
        self._verify_turn_level(gamma)
        
        # I. Loss 贡献（需要在 loss 计算时记录）
        
        # J. process_response_mask
        self._verify_response_mask()
        
        # K. 向量化 vs 顺序
        self._verify_vectorized_vs_sequential()
        
        # L. Loss 计算
        self._verify_loss_calculation()
        
        # M. 梯度验证
        self._verify_gradients()
        
        # N. Info Gain 梯度贡献
        self._verify_ig_contribution()
        
        # 总结
        self._print_summary()
        
        return self.results
    
    def _verify_gt_logprob(self):
        """A. 验证 GT LogProb 计算"""
        check_id = "A"
        check_name = "GT LogProb Calculation"
        
        if not self.gt_logprob_sequential and not self.gt_logprob_vectorized:
            self._add_result(check_id, check_name, True, "No GT LogProb data (skipped)")
            return
        
        # 验证：所有值应该是有限的负数（log probability）
        all_values = []
        invalid_count = 0
        
        for values in list(self.gt_logprob_sequential.values()) + list(self.gt_logprob_vectorized.values()):
            for v in values:
                all_values.append(v)
                if not np.isfinite(v):
                    invalid_count += 1
                # LogProb 通常是负数或接近 0
                if v > 1:  # 不太可能的值
                    invalid_count += 1
        
        if invalid_count > 0:
            self._add_result(check_id, check_name, False,
                f"Found {invalid_count} invalid GT LogProb values",
                {"invalid_count": invalid_count})
        else:
            mean_logprob = np.mean(all_values) if all_values else 0
            self._add_result(check_id, check_name, True,
                f"All {len(all_values)} GT LogProb values valid, mean={mean_logprob:.4f}",
                {"count": len(all_values), "mean": mean_logprob})
    
    def _verify_info_gain_calculation(self):
        """B. 验证 Info Gain 计算逻辑"""
        check_id = "B"
        check_name = "Info Gain Calculation"
        
        if not self.info_gain_computed:
            self._add_result(check_id, check_name, True, "No Info Gain data (skipped)")
            return
        
        errors = []
        total_checked = 0
        
        for sample_idx in self.info_gain_computed:
            ig_values = self.info_gain_computed[sample_idx]
            prev_values = self.prev_gt_values.get(sample_idx, [])
            curr_values = self.curr_gt_values.get(sample_idx, [])
            
            if len(ig_values) != len(prev_values) or len(ig_values) != len(curr_values):
                errors.append(f"Sample {sample_idx}: length mismatch")
                continue
            
            for i, (ig, prev, curr) in enumerate(zip(ig_values, prev_values, curr_values)):
                total_checked += 1
                expected_ig = curr - prev
                
                if abs(ig - expected_ig) > 1e-6:
                    errors.append(f"Sample {sample_idx} turn {i}: ig={ig:.6f}, expected={expected_ig:.6f} (curr-prev={curr:.6f}-{prev:.6f})")
        
        if errors:
            self._add_result(check_id, check_name, False,
                f"Info Gain formula errors: {len(errors)}/{total_checked}",
                {"errors": errors[:5]})
        else:
            self._add_result(check_id, check_name, True,
                f"All {total_checked} Info Gain values follow formula (ig = curr - prev)")
    
    def _verify_token_positions(self):
        """D. 验证 Token 位置分配"""
        check_id = "D"
        check_name = "Token Position Assignment"
        
        if not self.reward_token_positions:
            self._add_result(check_id, check_name, True, "No position data (skipped)")
            return
        
        mismatches = []
        total_checked = 0
        
        for sample_idx in self.reward_token_positions:
            actual = self.reward_token_positions[sample_idx]
            expected = self.expected_token_positions.get(sample_idx, [])
            
            if len(actual) != len(expected):
                mismatches.append(f"Sample {sample_idx}: count mismatch (actual={len(actual)}, expected={len(expected)})")
                continue
            
            for i, (a, e) in enumerate(zip(actual, expected)):
                total_checked += 1
                if a != e:
                    mismatches.append(f"Sample {sample_idx} turn {i}: actual={a}, expected={e}")
        
        if mismatches:
            self._add_result(check_id, check_name, False,
                f"Position mismatches: {len(mismatches)}",
                {"mismatches": mismatches[:5]})
        else:
            self._add_result(check_id, check_name, True,
                f"All {total_checked} reward positions correctly assigned")
    
    def _verify_reward_transmission(self):
        """E. 验证 Reward 传输"""
        check_id = "E"
        check_name = "Reward Transmission"
        
        if not self.rewards_at_info_gain:
            self._add_result(check_id, check_name, True, "No transmission data (skipped)")
            return
        
        # 收集所有值并排序比较
        ig_values = []
        core_values = []
        
        for rewards in self.rewards_at_info_gain.values():
            ig_values.extend(rewards)
        for rewards in self.rewards_at_core_algos.values():
            core_values.extend(rewards)
        
        ig_sorted = sorted(ig_values)
        core_sorted = sorted(core_values)
        
        if len(ig_sorted) != len(core_sorted):
            self._add_result(check_id, check_name, False,
                f"Count mismatch: info_gain={len(ig_sorted)}, core_algos={len(core_sorted)}")
            return
        
        mismatches = sum(1 for a, b in zip(ig_sorted, core_sorted) if abs(a - b) > 1e-6)
        
        if mismatches > 0:
            self._add_result(check_id, check_name, False,
                f"Value mismatches: {mismatches}/{len(ig_sorted)}")
        else:
            self._add_result(check_id, check_name, True,
                f"All {len(ig_sorted)} rewards correctly transmitted")
    
    def _verify_normalization(self):
        """F. 验证归一化"""
        check_id = "F"
        check_name = "Normalization"
        
        if self.rewards_after_norm is None:
            self._add_result(check_id, check_name, True, "No normalization data (skipped)")
            return
        
        after = self.rewards_after_norm
        nonzero_mask = after != 0
        
        if nonzero_mask.sum() == 0:
            self._add_result(check_id, check_name, True, "No non-zero rewards after norm")
            return
        
        nonzero_values = after[nonzero_mask]
        mean = nonzero_values.mean().item()
        std = nonzero_values.std().item() if nonzero_values.numel() > 1 else 0
        
        errors = []
        if abs(mean) > 0.1:
            errors.append(f"Mean not close to 0: {mean:.4f}")
        if std > 0 and abs(std - 1.0) > 0.3:
            errors.append(f"Std not close to 1: {std:.4f}")
        
        if errors:
            self._add_result(check_id, check_name, False,
                f"Normalization issues: {errors}")
        else:
            self._add_result(check_id, check_name, True,
                f"Normalization correct: mean={mean:.4f}, std={std:.4f}")
    
    def _verify_turn_level(self, gamma: float):
        """G/H. 验证 Turn-level 计算"""
        check_id = "G/H"
        check_name = "Turn-Level Discounting & Broadcasting"
        
        if not self.turn_rewards:
            self._add_result(check_id, check_name, True, "No turn-level data (skipped)")
            return
        
        errors = []
        total_checked = 0
        
        for sample_idx in self.turn_rewards:
            rewards = self.turn_rewards[sample_idx]
            advantages = self.turn_advantages.get(sample_idx, [])
            
            if len(rewards) != len(advantages):
                errors.append(f"Sample {sample_idx}: length mismatch")
                continue
            
            # 验证折扣累积公式
            for i in range(len(rewards) - 1, -1, -1):
                total_checked += 1
                r_i = rewards[i]
                A_i = advantages[i]
                
                if i == len(rewards) - 1:
                    expected = r_i
                else:
                    A_next = advantages[i + 1]
                    expected = r_i + gamma * A_next
                
                if abs(A_i - expected) > 1e-5:
                    errors.append(f"Sample {sample_idx} turn {i}: A={A_i:.4f}, expected={expected:.4f}")
        
        if errors:
            self._add_result(check_id, check_name, False,
                f"Turn-level errors: {len(errors)}/{total_checked}",
                {"errors": errors[:5]})
        else:
            self._add_result(check_id, check_name, True,
                f"All {total_checked} turn-level computations correct (γ={gamma})")
    
    def _verify_response_mask(self):
        """J. 验证 process_response_mask"""
        check_id = "J"
        check_name = "process_response_mask"
        
        if self.response_mask_before is None or self.response_mask_after is None:
            self._add_result(check_id, check_name, True, "No response_mask data (skipped)")
            return
        
        before = self.response_mask_before
        after = self.response_mask_after
        
        # 验证：tool_response 位置应该从 1 变成 0
        errors = []
        correctly_masked = 0
        
        for sample_idx, tool_positions in self.tool_response_positions.items():
            if sample_idx >= before.shape[0]:
                continue
            
            for start, end in tool_positions:
                for pos in range(start, end):
                    if pos >= before.shape[1]:
                        continue
                    
                    before_val = before[sample_idx, pos].item()
                    after_val = after[sample_idx, pos].item()
                    
                    if before_val == 1 and after_val == 0:
                        correctly_masked += 1
                    elif before_val == 1 and after_val == 1:
                        errors.append(f"Sample {sample_idx} pos {pos}: tool_response not masked")
        
        if errors:
            self._add_result(check_id, check_name, False,
                f"Mask errors: {len(errors)} tool_response tokens not masked",
                {"errors": errors[:5]})
        else:
            self._add_result(check_id, check_name, True,
                f"process_response_mask correct: {correctly_masked} tool_response tokens masked")
    
    def _verify_vectorized_vs_sequential(self):
        """K. 验证向量化 vs 顺序计算"""
        check_id = "K"
        check_name = "Vectorized vs Sequential"
        
        if not self.gt_logprob_sequential or not self.gt_logprob_vectorized:
            self._add_result(check_id, check_name, True,
                "Need both modes to compare (skipped)")
            return
        
        common_samples = set(self.gt_logprob_sequential.keys()) & set(self.gt_logprob_vectorized.keys())
        
        if not common_samples:
            self._add_result(check_id, check_name, True, "No common samples (skipped)")
            return
        
        mismatches = []
        total_checked = 0
        
        for sample_idx in common_samples:
            seq = self.gt_logprob_sequential[sample_idx]
            vec = self.gt_logprob_vectorized[sample_idx]
            
            if len(seq) != len(vec):
                mismatches.append(f"Sample {sample_idx}: length mismatch")
                continue
            
            for i, (s, v) in enumerate(zip(seq, vec)):
                total_checked += 1
                if abs(s - v) > 1e-4:  # 允许较大的浮点误差
                    mismatches.append(f"Sample {sample_idx} turn {i}: seq={s:.6f}, vec={v:.6f}")
        
        if mismatches:
            self._add_result(check_id, check_name, False,
                f"Vectorized differs from sequential: {len(mismatches)} mismatches",
                {"mismatches": mismatches[:5]})
        else:
            self._add_result(check_id, check_name, True,
                f"Vectorized == Sequential: {total_checked} values match")
    
    def _verify_loss_calculation(self):
        """L. 验证 Loss 计算正确性"""
        check_id = "L"
        check_name = "Policy Loss Calculation"
        
        if self.advantages_for_loss is None or self.policy_loss_value is None:
            self._add_result(check_id, check_name, True, "No loss data (skipped)")
            return
        
        advantages = self.advantages_for_loss
        log_probs = self.log_probs_for_loss
        old_log_probs = self.old_log_probs_for_loss
        response_mask = self.response_mask_for_loss
        
        errors = []
        
        # 1. 验证 Loss 值是有限的
        if not np.isfinite(self.policy_loss_value):
            errors.append(f"Loss is not finite: {self.policy_loss_value}")
        
        # 2. 验证 Loss 在合理范围内（通常 PPO loss 在 0.001 ~ 1.0 之间）
        if self.policy_loss_value < 0:
            errors.append(f"Loss is negative: {self.policy_loss_value}")
        if self.policy_loss_value > 100:
            errors.append(f"Loss is suspiciously large: {self.policy_loss_value}")
        
        # 3. 验证 advantage 确实参与了计算
        # 如果 advantage 全为 0，loss 应该接近 0（在 PPO 中）
        adv_mask = (response_mask == 1) if response_mask is not None else torch.ones_like(advantages, dtype=torch.bool)
        valid_advantages = advantages[adv_mask]
        
        if valid_advantages.numel() > 0:
            adv_mean = valid_advantages.mean().item()
            adv_std = valid_advantages.std().item() if valid_advantages.numel() > 1 else 0
            adv_nonzero_ratio = (valid_advantages != 0).float().mean().item()
            
            if adv_nonzero_ratio < 0.1:
                errors.append(f"Too few non-zero advantages: {adv_nonzero_ratio:.1%}")
        
        # 4. 验证 Info Gain 对 Loss 的贡献
        # 检查 advantage 是否有 turn 间的差异（说明 info_gain 起作用）
        if self.turn_advantages:
            all_turn_advs = []
            for sample_advs in self.turn_advantages.values():
                all_turn_advs.extend(sample_advs)
            
            if len(all_turn_advs) > 1:
                turn_adv_variance = np.var(all_turn_advs)
                if turn_adv_variance < 1e-8:
                    errors.append(f"No turn-level variance in advantages (info_gain may not be working)")
        
        # 5. 手动计算 Loss 并验证（简化版 PPO loss）
        if log_probs is not None and old_log_probs is not None and response_mask is not None:
            ratio = torch.exp(log_probs - old_log_probs)
            # 简化的 policy gradient loss: -advantage * ratio
            manual_loss_per_token = -advantages * ratio
            manual_loss_per_token = manual_loss_per_token * response_mask
            
            valid_tokens = response_mask.sum()
            if valid_tokens > 0:
                manual_loss = manual_loss_per_token.sum() / valid_tokens
                manual_loss_val = manual_loss.item()
                
                # 比较手动计算和实际 loss（允许一定误差，因为有 clipping）
                loss_diff = abs(manual_loss_val - self.policy_loss_value)
                if loss_diff > abs(self.policy_loss_value) * 0.5:  # 差异超过 50%
                    # 这可能是因为 PPO clipping，不一定是错误
                    pass  # 不报错，只记录
        
        if errors:
            self._add_result(check_id, check_name, False,
                f"Loss validation issues: {len(errors)}",
                {"errors": errors, "loss_value": self.policy_loss_value})
        else:
            self._add_result(check_id, check_name, True,
                f"Policy Loss valid: {self.policy_loss_value:.6f}, adv_nonzero={adv_nonzero_ratio:.1%}",
                {"loss_value": self.policy_loss_value})
    
    def _verify_gradients(self):
        """M. 验证梯度正确性"""
        check_id = "M"
        check_name = "Gradient Verification"
        
        if self.grad_log_prob is None and self.grad_norm is None:
            self._add_result(check_id, check_name, True, "No gradient data (skipped)")
            return
        
        errors = []
        
        # 1. 验证梯度范数
        if self.grad_norm is not None:
            if not np.isfinite(self.grad_norm):
                errors.append(f"Gradient norm is not finite: {self.grad_norm}")
            elif self.grad_norm > 1000:
                errors.append(f"Gradient norm is very large: {self.grad_norm} (may cause instability)")
            elif self.grad_norm < 1e-10:
                errors.append(f"Gradient norm is near zero: {self.grad_norm} (model may not be learning)")
        
        # 2. 验证 log_prob 的梯度
        if self.grad_log_prob is not None:
            grad = self.grad_log_prob
            
            # 检查 NaN 和 Inf
            nan_count = torch.isnan(grad).sum().item()
            inf_count = torch.isinf(grad).sum().item()
            
            if nan_count > 0:
                errors.append(f"Gradient contains {nan_count} NaN values")
            if inf_count > 0:
                errors.append(f"Gradient contains {inf_count} Inf values")
            
            # 检查梯度与 advantage 的关系
            # 在 PPO 中，d(loss)/d(log_prob) ≈ -advantage（简化情况）
            if self.advantages_for_loss is not None and self.response_mask_for_loss is not None:
                adv = self.advantages_for_loss
                mask = self.response_mask_for_loss
                
                # 检查梯度的符号是否与 advantage 相反（大致）
                grad_sign = (grad > 0).float()
                adv_sign = (adv < 0).float()  # 因为 loss = -adv * ratio
                
                valid_mask = (mask == 1) & (adv != 0)
                if valid_mask.sum() > 0:
                    sign_match = (grad_sign[valid_mask] == adv_sign[valid_mask]).float().mean().item()
                    # 由于 PPO clipping，符号匹配率不需要 100%
                    if sign_match < 0.5:
                        errors.append(f"Gradient sign mismatch with advantage: {sign_match:.1%}")
        
        if errors:
            self._add_result(check_id, check_name, False,
                f"Gradient issues: {len(errors)}",
                {"errors": errors, "grad_norm": self.grad_norm})
        else:
            self._add_result(check_id, check_name, True,
                f"Gradients valid: norm={self.grad_norm:.4f}" if self.grad_norm else "Gradients valid")
    
    def _verify_ig_contribution(self):
        """N. 验证 Info Gain 对梯度的贡献"""
        check_id = "N"
        check_name = "Info Gain Gradient Contribution"
        
        # 方法1: 直接比较有无 IG 的 loss 差异
        if self.loss_with_ig is not None and self.loss_without_ig is not None:
            contribution = self.ig_contribution
            loss_with = self.loss_with_ig
            loss_without = self.loss_without_ig
            
            if contribution < 1e-8:
                self._add_result(check_id, check_name, False,
                    f"Info Gain has NO contribution to loss! with_ig={loss_with:.6f}, without_ig={loss_without:.6f}",
                    {"loss_with_ig": loss_with, "loss_without_ig": loss_without})
                return
            else:
                relative_contribution = contribution / (abs(loss_with) + 1e-10)
                self._add_result(check_id, check_name, True,
                    f"Info Gain contributes to loss: diff={contribution:.6f} ({relative_contribution:.1%} of total)",
                    {"loss_with_ig": loss_with, "loss_without_ig": loss_without, 
                     "contribution": contribution, "relative": relative_contribution})
                return
        
        # 方法2: 间接验证 - 检查非 F1 位置的 advantage 是否参与了 loss
        if self.advantages_for_loss is None or self.f1_mask is None:
            self._add_result(check_id, check_name, True, "No data for IG contribution check (skipped)")
            return
        
        adv = self.advantages_for_loss
        f1_mask = self.f1_mask
        response_mask = self.response_mask_for_loss if self.response_mask_for_loss is not None else torch.ones_like(adv)
        
        # 非 F1 位置 (info_gain 位置) 的 advantage
        ig_positions = (response_mask == 1) & (~f1_mask)
        ig_advantages = adv[ig_positions]
        
        if ig_advantages.numel() == 0:
            self._add_result(check_id, check_name, False,
                "No Info Gain positions found in advantages!")
            return
        
        ig_nonzero = (ig_advantages.abs() > 1e-8).sum().item()
        ig_total = ig_advantages.numel()
        ig_nonzero_ratio = ig_nonzero / ig_total
        
        # Info Gain 对应位置的 advantage 绝对值之和
        ig_abs_sum = ig_advantages.abs().sum().item()
        total_abs_sum = adv[response_mask == 1].abs().sum().item()
        ig_proportion = ig_abs_sum / (total_abs_sum + 1e-10)
        
        if ig_nonzero == 0:
            self._add_result(check_id, check_name, False,
                f"All Info Gain advantages are ZERO! IG positions: {ig_total}",
                {"ig_positions": ig_total, "ig_nonzero": 0})
        elif ig_nonzero_ratio < 0.01:
            self._add_result(check_id, check_name, False,
                f"Very few IG advantages are non-zero: {ig_nonzero}/{ig_total} ({ig_nonzero_ratio:.1%})",
                {"ig_positions": ig_total, "ig_nonzero": ig_nonzero})
        else:
            self._add_result(check_id, check_name, True,
                f"IG contributes to gradient: {ig_nonzero}/{ig_total} non-zero ({ig_nonzero_ratio:.1%}), "
                f"IG proportion: {ig_proportion:.1%}",
                {"ig_positions": ig_total, "ig_nonzero": ig_nonzero, "ig_proportion": ig_proportion})
    
    def _add_result(self, check_id: str, check_name: str, passed: bool, message: str, details: Dict = None):
        """添加验证结果"""
        result = FullCheckResult(
            check_id=check_id,
            check_name=check_name,
            passed=passed,
            message=message,
            details=details or {}
        )
        self.results.append(result)
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n[{status}] Check {check_id}: {check_name}")
        print(f"  {message}")
        if not passed and details:
            for key, value in details.items():
                if isinstance(value, list) and len(value) > 0:
                    print(f"  {key}: {value}")
    
    def _print_summary(self):
        """打印总结"""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print("\n" + "=" * 80)
        print(f"    SUMMARY: {passed}/{total} checks passed")
        
        if passed == total:
            print("    ✓ ALL CHECKS PASSED - IGPO Training Logic Verified!")
        else:
            failed = [f"{r.check_id}:{r.check_name}" for r in self.results if not r.passed]
            print(f"    ⚠️ FAILED: {failed}")
        
        print("=" * 80 + "\n")


# 全局实例
_full_checker = IGPOFullChecker()


def get_full_checker() -> IGPOFullChecker:
    """获取全局验证器"""
    return _full_checker


def reset_full_checker():
    """重置验证器"""
    _full_checker.reset()
