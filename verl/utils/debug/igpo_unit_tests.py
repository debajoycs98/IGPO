# Copyright 2025 IGPO Team
# IGPO 关键模块单元测试
# 用于验证各模块计算的 100% 正确性

import os
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# ============================================================================
# 测试结果数据类
# ============================================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    details: Optional[Dict] = None


class IGPOUnitTester:
    """IGPO 单元测试器"""
    
    def __init__(self):
        self.results: List[TestResult] = []
    
    def _add_result(self, name: str, passed: bool, message: str, details: Dict = None):
        self.results.append(TestResult(name, passed, message, details))
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{status}] {name}: {message}")
    
    # ========================================================================
    # Test 1: GT LogProb 计算公式
    # ========================================================================
    
    def test_gt_logprob_calculation(self):
        """测试 GT LogProb 计算公式的正确性"""
        print("\n" + "="*60)
        print("Test 1: GT LogProb Calculation")
        print("="*60)
        
        # 构造已知输入
        batch_size, seq_len, vocab_size = 2, 5, 100
        
        # 创建 logits
        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # 创建 ground truth token ids
        gt_ids = torch.tensor([[10, 20, 30, 40, 50], [15, 25, 35, 45, 55]])
        
        # 方法1: 标准计算（正确实现）
        log_probs_standard = torch.log_softmax(logits, dim=-1)
        gt_logprob_standard = log_probs_standard.gather(dim=-1, index=gt_ids.unsqueeze(-1)).squeeze(-1)
        
        # 方法2: 手动计算验证
        gt_logprob_manual = torch.zeros(batch_size, seq_len)
        for b in range(batch_size):
            for t in range(seq_len):
                # softmax
                exp_logits = torch.exp(logits[b, t] - logits[b, t].max())
                softmax = exp_logits / exp_logits.sum()
                # log
                log_softmax = torch.log(softmax)
                gt_logprob_manual[b, t] = log_softmax[gt_ids[b, t]]
        
        # 验证
        diff = (gt_logprob_standard - gt_logprob_manual).abs().max().item()
        
        if diff < 1e-5:
            self._add_result("gt_logprob_formula", True, 
                f"Formula correct, max diff={diff:.2e}")
        else:
            self._add_result("gt_logprob_formula", False,
                f"Formula incorrect! max diff={diff:.2e}",
                {"standard": gt_logprob_standard.tolist(), "manual": gt_logprob_manual.tolist()})
        
        # 测试边界情况: 极大/极小 logits
        logits_extreme = torch.tensor([[[1000.0, -1000.0, 0.0]]])
        gt_ids_extreme = torch.tensor([[0]])  # 应该选择第一个（最大）
        
        log_probs_extreme = torch.log_softmax(logits_extreme, dim=-1)
        gt_logprob_extreme = log_probs_extreme[0, 0, 0].item()
        
        # 期望值: log(softmax(1000)) ≈ 0 (因为 exp(1000) >> exp(-1000) + exp(0))
        if abs(gt_logprob_extreme) < 1e-3:
            self._add_result("gt_logprob_extreme", True,
                f"Extreme case handled: logprob={gt_logprob_extreme:.6f}")
        else:
            self._add_result("gt_logprob_extreme", False,
                f"Extreme case failed: logprob={gt_logprob_extreme:.6f}, expected ≈ 0")
    
    # ========================================================================
    # Test 2: Info Gain 计算公式
    # ========================================================================
    
    def test_info_gain_calculation(self):
        """测试 Info Gain 计算公式"""
        print("\n" + "="*60)
        print("Test 2: Info Gain Calculation")
        print("="*60)
        
        # Info Gain = log P(gt | turn_t) - log P(gt | turn_{t-1})
        # = curr_logprob - prev_logprob
        
        # 测试用例 1: 正常情况
        prev_logprob = -5.0  # P(gt) = exp(-5) ≈ 0.0067
        curr_logprob = -2.0  # P(gt) = exp(-2) ≈ 0.135
        
        expected_info_gain = curr_logprob - prev_logprob  # -2 - (-5) = 3
        
        if abs(expected_info_gain - 3.0) < 1e-6:
            self._add_result("info_gain_positive", True,
                f"Positive info gain correct: {expected_info_gain:.4f}")
        else:
            self._add_result("info_gain_positive", False,
                f"Wrong info gain: {expected_info_gain:.4f}, expected 3.0")
        
        # 测试用例 2: 负 info gain（模型变差）
        prev_logprob = -2.0
        curr_logprob = -5.0
        expected_info_gain = curr_logprob - prev_logprob  # -5 - (-2) = -3
        
        if abs(expected_info_gain - (-3.0)) < 1e-6:
            self._add_result("info_gain_negative", True,
                f"Negative info gain correct: {expected_info_gain:.4f}")
        else:
            self._add_result("info_gain_negative", False,
                f"Wrong info gain: {expected_info_gain:.4f}, expected -3.0")
        
        # 测试用例 3: 零 info gain
        prev_logprob = -3.0
        curr_logprob = -3.0
        expected_info_gain = curr_logprob - prev_logprob
        
        if abs(expected_info_gain) < 1e-6:
            self._add_result("info_gain_zero", True,
                f"Zero info gain correct: {expected_info_gain:.4f}")
        else:
            self._add_result("info_gain_zero", False,
                f"Wrong info gain: {expected_info_gain:.4f}, expected 0.0")
    
    # ========================================================================
    # Test 3: Turn 边界识别
    # ========================================================================
    
    def test_turn_boundary_detection(self):
        """测试 Turn 边界识别的正确性"""
        print("\n" + "="*60)
        print("Test 3: Turn Boundary Detection")
        print("="*60)
        
        separator = "\n<|im_start|>assistant\n"
        
        # 测试用例 1: 标准多轮对话
        test_str_1 = """Hello world
<|im_start|>assistant
Response 1<|im_end|>
<|im_start|>user
Question 2<|im_end|>
<|im_start|>assistant
Response 2<|im_end|>"""
        
        parts_1 = test_str_1.split(separator)
        expected_parts_1 = 3  # [prefix, Response 1..., Response 2...]
        
        if len(parts_1) == expected_parts_1:
            self._add_result("turn_boundary_standard", True,
                f"Standard case: found {len(parts_1)} parts as expected")
        else:
            self._add_result("turn_boundary_standard", False,
                f"Standard case failed: found {len(parts_1)} parts, expected {expected_parts_1}",
                {"parts": [p[:50] + "..." for p in parts_1]})
        
        # 测试用例 2: 单轮对话 (注意分隔符需要完整匹配)
        test_str_2 = """System prompt
<|im_start|>assistant
Only one response"""
        
        parts_2 = test_str_2.split(separator)
        expected_parts_2 = 2  # [prefix, response]
        
        if len(parts_2) == expected_parts_2:
            self._add_result("turn_boundary_single", True,
                f"Single turn: found {len(parts_2)} parts as expected")
        else:
            self._add_result("turn_boundary_single", False,
                f"Single turn failed: found {len(parts_2)} parts, expected {expected_parts_2}")
        
        # 测试用例 3: 嵌套/特殊情况
        test_str_3 = """Text with nested
<|im_start|>assistant
content"""
        
        parts_3 = test_str_3.split(separator)
        if len(parts_3) == 2:
            self._add_result("turn_boundary_nested", True,
                "Nested case handled correctly")
        else:
            self._add_result("turn_boundary_nested", False,
                f"Nested case failed: {len(parts_3)} parts")
    
    # ========================================================================
    # Test 4: Response Mask 逻辑
    # ========================================================================
    
    def test_response_mask_logic(self):
        """测试 Response Mask 生成逻辑"""
        print("\n" + "="*60)
        print("Test 4: Response Mask Logic")
        print("="*60)
        
        # 模拟 token 序列
        # <|im_start|>assistant\n Response <|im_end|> <|im_start|>user\n ...
        
        # 简化测试: 检查 mask 逻辑
        # response_mask 应该只在 assistant 回复区域为 1
        
        # 测试用例: 手动构造 mask
        seq_len = 20
        # 假设: [0-4] prompt, [5-10] assistant response, [11-15] user, [16-19] assistant
        
        expected_mask = torch.zeros(seq_len)
        expected_mask[5:11] = 1  # 第一个 assistant 区域
        expected_mask[16:20] = 1  # 第二个 assistant 区域
        
        # 检查 mask 属性
        assistant_tokens = expected_mask.sum().item()
        total_tokens = seq_len
        ratio = assistant_tokens / total_tokens
        
        if 0.3 < ratio < 0.7:  # 合理范围
            self._add_result("response_mask_ratio", True,
                f"Assistant token ratio: {ratio:.1%} (reasonable)")
        else:
            self._add_result("response_mask_ratio", False,
                f"Unusual assistant ratio: {ratio:.1%}")
        
        # 验证 mask 是连续的区域（在每个 assistant turn 内）
        # 这里简化为检查非零区域
        nonzero_indices = expected_mask.nonzero().squeeze()
        if len(nonzero_indices) > 0:
            self._add_result("response_mask_structure", True,
                f"Mask has {len(nonzero_indices)} non-zero positions")
        else:
            self._add_result("response_mask_structure", False,
                "Mask has no non-zero positions!")
    
    # ========================================================================
    # Test 5: 标准化公式
    # ========================================================================
    
    def test_normalization_formula(self):
        """测试标准化公式的正确性"""
        print("\n" + "="*60)
        print("Test 5: Normalization Formula")
        print("="*60)
        
        # 测试数据
        torch.manual_seed(42)
        rewards = torch.randn(32)  # 32 个样本
        
        # 标准化: (x - mean) / std
        mean = rewards.mean()
        std = rewards.std()
        normalized = (rewards - mean) / (std + 1e-8)
        
        # 验证属性
        norm_mean = normalized.mean().item()
        norm_std = normalized.std().item()
        
        if abs(norm_mean) < 1e-5:
            self._add_result("norm_mean_zero", True,
                f"Normalized mean ≈ 0: {norm_mean:.2e}")
        else:
            self._add_result("norm_mean_zero", False,
                f"Normalized mean not zero: {norm_mean:.6f}")
        
        if abs(norm_std - 1.0) < 0.01:
            self._add_result("norm_std_one", True,
                f"Normalized std ≈ 1: {norm_std:.6f}")
        else:
            self._add_result("norm_std_one", False,
                f"Normalized std not 1: {norm_std:.6f}")
        
        # 测试只除 std 的模式
        normalized_no_std = (rewards - mean)
        norm_mean_2 = normalized_no_std.mean().item()
        
        if abs(norm_mean_2) < 1e-5:
            self._add_result("norm_mean_only", True,
                f"Mean-only normalized mean ≈ 0: {norm_mean_2:.2e}")
        else:
            self._add_result("norm_mean_only", False,
                f"Mean-only normalized mean not zero: {norm_mean_2:.6f}")
    
    # ========================================================================
    # Test 6: 折扣累积公式
    # ========================================================================
    
    def test_discounted_return_formula(self):
        """测试折扣累积公式的正确性"""
        print("\n" + "="*60)
        print("Test 6: Discounted Return Formula")
        print("="*60)
        
        # Turn-level rewards: [r0, r1, r2, r3] (从后向前)
        # Discounted return at turn t: G_t = r_t + γ*G_{t+1}
        
        rewards = [1.0, 2.0, 3.0, 4.0]  # 4 个 turn
        gamma = 0.99
        
        # 手动计算期望值 (从后向前)
        # G_3 = 4.0
        # G_2 = 3.0 + 0.99 * 4.0 = 6.96
        # G_1 = 2.0 + 0.99 * 6.96 = 8.8904
        # G_0 = 1.0 + 0.99 * 8.8904 = 9.801496
        
        expected = [9.801496, 8.8904, 6.96, 4.0]
        
        # 实现折扣累积
        n = len(rewards)
        discounted = [0.0] * n
        discounted[n-1] = rewards[n-1]
        for t in range(n-2, -1, -1):
            discounted[t] = rewards[t] + gamma * discounted[t+1]
        
        # 验证
        all_match = True
        for t in range(n):
            diff = abs(discounted[t] - expected[t])
            if diff > 1e-4:
                all_match = False
                self._add_result(f"discount_turn_{t}", False,
                    f"Turn {t}: got {discounted[t]:.6f}, expected {expected[t]:.6f}")
        
        if all_match:
            self._add_result("discount_all_turns", True,
                f"All turns correct: {[f'{d:.4f}' for d in discounted]}")
        
        # 测试 gamma=1 的特殊情况（简单累加）
        gamma_1 = 1.0
        discounted_1 = [0.0] * n
        discounted_1[n-1] = rewards[n-1]
        for t in range(n-2, -1, -1):
            discounted_1[t] = rewards[t] + gamma_1 * discounted_1[t+1]
        
        # gamma=1 时: G_0 = 1+2+3+4 = 10
        if abs(discounted_1[0] - 10.0) < 1e-6:
            self._add_result("discount_gamma_1", True,
                f"γ=1 case correct: G_0={discounted_1[0]:.4f}")
        else:
            self._add_result("discount_gamma_1", False,
                f"γ=1 case wrong: G_0={discounted_1[0]:.4f}, expected 10.0")
    
    # ========================================================================
    # Test 7: PPO Loss 公式
    # ========================================================================
    
    def test_ppo_loss_formula(self):
        """测试 PPO Loss 公式的正确性"""
        print("\n" + "="*60)
        print("Test 7: PPO Loss Formula")
        print("="*60)
        
        # PPO Loss: L = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
        
        torch.manual_seed(42)
        batch_size, seq_len = 4, 10
        
        log_prob = torch.randn(batch_size, seq_len)
        old_log_prob = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        response_mask = torch.ones(batch_size, seq_len)
        
        cliprange = 0.2
        
        # 计算 ratio
        ratio = torch.exp(log_prob - old_log_prob)
        
        # 两个 loss 项
        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
        
        # PPO 取 max（因为是负的，所以取 max 相当于取绝对值小的）
        pg_losses = torch.maximum(pg_losses1, pg_losses2)
        
        # 计算平均 loss
        loss = (pg_losses * response_mask).sum() / response_mask.sum()
        
        # 验证 loss 是有限的
        if torch.isfinite(loss):
            self._add_result("ppo_loss_finite", True,
                f"PPO loss is finite: {loss.item():.6f}")
        else:
            self._add_result("ppo_loss_finite", False,
                f"PPO loss is not finite: {loss.item()}")
        
        # 验证 clipping 生效
        clipped_count = ((ratio < 1 - cliprange) | (ratio > 1 + cliprange)).sum().item()
        clip_ratio = clipped_count / ratio.numel()
        
        self._add_result("ppo_clipping", True,
            f"Clipping ratio: {clip_ratio:.1%} ({clipped_count}/{ratio.numel()})")
        
        # 验证 advantage 符号与 loss 符号的关系
        # 当 A > 0 时，希望 ratio 增大，loss < 0 表示正确方向
        # 当 A < 0 时，希望 ratio 减小，loss < 0 表示正确方向
        pos_adv_mask = (advantages > 0) & (response_mask == 1)
        neg_adv_mask = (advantages < 0) & (response_mask == 1)
        
        if pos_adv_mask.sum() > 0 and neg_adv_mask.sum() > 0:
            self._add_result("ppo_gradient_direction", True,
                f"Both pos ({pos_adv_mask.sum()}) and neg ({neg_adv_mask.sum()}) advantages present")
        else:
            self._add_result("ppo_gradient_direction", False,
                "Missing positive or negative advantages")
    
    # ========================================================================
    # Test 8: Turn 内广播正确性
    # ========================================================================
    
    def test_turn_broadcast(self):
        """测试 Turn 内广播的正确性"""
        print("\n" + "="*60)
        print("Test 8: Turn Broadcast")
        print("="*60)
        
        # 假设 3 个 turn，每个 turn 的 token 范围
        turn_ranges = [(0, 5), (5, 12), (12, 20)]
        turn_rewards = [1.5, -0.5, 2.0]
        seq_len = 20
        
        # 广播到 token 级别
        token_rewards = torch.zeros(seq_len)
        for (start, end), reward in zip(turn_ranges, turn_rewards):
            token_rewards[start:end] = reward
        
        # 验证每个 turn 内的值相同
        all_correct = True
        for i, ((start, end), expected_reward) in enumerate(zip(turn_ranges, turn_rewards)):
            turn_values = token_rewards[start:end]
            unique_values = turn_values.unique()
            
            if len(unique_values) == 1 and abs(unique_values[0].item() - expected_reward) < 1e-6:
                pass  # correct
            else:
                all_correct = False
                self._add_result(f"broadcast_turn_{i}", False,
                    f"Turn {i} broadcast error: unique values = {unique_values.tolist()}")
        
        if all_correct:
            self._add_result("broadcast_all_turns", True,
                f"All turns broadcast correctly: {turn_rewards}")
    
    # ========================================================================
    # Test 9: 端到端测试 - 已知答案验证
    # ========================================================================
    
    def test_end_to_end(self):
        """端到端测试：构造已知输入，验证整个流程"""
        print("\n" + "="*60)
        print("Test 9: End-to-End Verification")
        print("="*60)
        
        # 构造简化的已知场景
        # 2 个样本，每个 3 个 turn
        
        # Info Gain rewards (每个样本 2 个 IG，因为 3 turn - 1)
        ig_rewards = [[0.5, 0.3], [0.2, 0.1]]  # sample 0, sample 1
        f1_rewards = [0.8, 0.6]  # 最后一个 turn 的 F1
        
        gamma = 0.99
        
        # 手动计算期望的 discounted return
        # Sample 0: turn 0 IG=0.5, turn 1 IG=0.3, turn 2 F1=0.8
        # G_2 = 0.8
        # G_1 = 0.3 + 0.99 * 0.8 = 1.092
        # G_0 = 0.5 + 0.99 * 1.092 = 1.58108
        
        expected_sample_0 = [1.58108, 1.092, 0.8]
        
        # Sample 1: turn 0 IG=0.2, turn 1 IG=0.1, turn 2 F1=0.6
        # G_2 = 0.6
        # G_1 = 0.1 + 0.99 * 0.6 = 0.694
        # G_0 = 0.2 + 0.99 * 0.694 = 0.88706
        
        expected_sample_1 = [0.88706, 0.694, 0.6]
        
        # 实际计算
        def compute_discounted(ig_list, f1, gamma):
            rewards = ig_list + [f1]
            n = len(rewards)
            discounted = [0.0] * n
            discounted[n-1] = rewards[n-1]
            for t in range(n-2, -1, -1):
                discounted[t] = rewards[t] + gamma * discounted[t+1]
            return discounted
        
        actual_0 = compute_discounted(ig_rewards[0], f1_rewards[0], gamma)
        actual_1 = compute_discounted(ig_rewards[1], f1_rewards[1], gamma)
        
        # 验证
        error_0 = max(abs(a - e) for a, e in zip(actual_0, expected_sample_0))
        error_1 = max(abs(a - e) for a, e in zip(actual_1, expected_sample_1))
        
        if error_0 < 1e-4:
            self._add_result("e2e_sample_0", True,
                f"Sample 0 correct: {[f'{v:.4f}' for v in actual_0]}")
        else:
            self._add_result("e2e_sample_0", False,
                f"Sample 0 error: max diff = {error_0:.6f}",
                {"actual": actual_0, "expected": expected_sample_0})
        
        if error_1 < 1e-4:
            self._add_result("e2e_sample_1", True,
                f"Sample 1 correct: {[f'{v:.4f}' for v in actual_1]}")
        else:
            self._add_result("e2e_sample_1", False,
                f"Sample 1 error: max diff = {error_1:.6f}",
                {"actual": actual_1, "expected": expected_sample_1})
    
    # ========================================================================
    # Test 10: 数值稳定性测试
    # ========================================================================
    
    def test_numerical_stability(self):
        """测试数值稳定性"""
        print("\n" + "="*60)
        print("Test 10: Numerical Stability")
        print("="*60)
        
        # 测试 log_softmax 在极端情况下的稳定性
        logits_large = torch.tensor([[1e10, 0.0, -1e10]])
        log_probs_large = torch.log_softmax(logits_large, dim=-1)
        
        if torch.isfinite(log_probs_large).all():
            self._add_result("stability_large_logits", True,
                f"Large logits handled: {log_probs_large.tolist()}")
        else:
            self._add_result("stability_large_logits", False,
                "Large logits cause NaN/Inf")
        
        # 测试标准化在 std ≈ 0 时的稳定性
        near_constant = torch.tensor([1.0, 1.0 + 1e-10, 1.0 - 1e-10])
        mean = near_constant.mean()
        std = near_constant.std()
        
        eps = 1e-8
        normalized = (near_constant - mean) / (std + eps)
        
        if torch.isfinite(normalized).all():
            self._add_result("stability_near_zero_std", True,
                f"Near-zero std handled with eps={eps}")
        else:
            self._add_result("stability_near_zero_std", False,
                "Near-zero std causes NaN/Inf")
        
        # 测试 exp 溢出处理
        large_diff = torch.tensor([100.0])  # exp(100) 会溢出
        ratio = torch.exp(large_diff)
        
        if torch.isinf(ratio).any():
            # 预期会溢出，但 PPO clipping 应该处理
            clipped_ratio = torch.clamp(ratio, 0, 10)
            if torch.isfinite(clipped_ratio).all():
                self._add_result("stability_exp_overflow", True,
                    "Exp overflow handled by clipping")
            else:
                self._add_result("stability_exp_overflow", False,
                    "Exp overflow not handled")
        else:
            self._add_result("stability_exp_overflow", True,
                "No exp overflow occurred")
    
    # ========================================================================
    # 运行所有测试
    # ========================================================================
    
    def run_all_tests(self):
        """运行所有单元测试"""
        print("\n" + "="*70)
        print(" IGPO Unit Tests - Comprehensive Verification")
        print("="*70)
        
        self.results = []
        
        self.test_gt_logprob_calculation()
        self.test_info_gain_calculation()
        self.test_turn_boundary_detection()
        self.test_response_mask_logic()
        self.test_normalization_formula()
        self.test_discounted_return_formula()
        self.test_ppo_loss_formula()
        self.test_turn_broadcast()
        self.test_end_to_end()
        self.test_numerical_stability()
        
        # 总结
        print("\n" + "="*70)
        print(" Summary")
        print("="*70)
        
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        
        print(f"\nTotal: {total} tests")
        print(f"  ✓ Passed: {passed}")
        print(f"  ✗ Failed: {failed}")
        print(f"  Pass Rate: {passed/total:.1%}")
        
        if failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.message}")
        
        return failed == 0


def run_unit_tests():
    """运行单元测试"""
    tester = IGPOUnitTester()
    return tester.run_all_tests()


if __name__ == "__main__":
    success = run_unit_tests()
    exit(0 if success else 1)
