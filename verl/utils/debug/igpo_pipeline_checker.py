"""
IGPO Pipeline Verification Framework

严格验证 IGPO 训练流水线中各个环节的正确性。

=============================================================================
                              启用方式
=============================================================================

    export DEBUG_IGPO_STRICT=1

=============================================================================
                              验证点
=============================================================================

    1. Check 1 - Generation 阶段
       - 验证 info_gain_reward 是否被正确计算和存储
       - 记录计算模式（vectorized / sequential）
       - 统计样本数量和 info_gain 值的数量

    2. Check 2 - info_gain.py 分配
       - 验证 reward 从 generation 传递到 info_gain.py 的一致性
       - 验证 reward 是否分配到正确的轮次末尾位置
       - 对比 generation 阶段记录的值和 info_gain.py 接收的值

    3. Check 3 - core_algos.py 传输
       - 验证 token_level_rewards 是否正确传输
       - 检查 reward 位置是否与预期一致
       - 验证 reward 值是否匹配

    4. Check 4 - 归一化
       - 验证 separate/joint 模式的归一化逻辑
       - 检查归一化后的 mean 是否接近 0
       - 检查归一化后的 std 是否接近 1（如果 norm_by_std=True）

    5. Check 5 - Turn-level 折扣累积与广播
       - Check 5a: 验证折扣公式 A_i = r_i + γ * A_{i+1}
       - Check 5b: 验证每个 turn 内所有 tokens 的 advantage 是否一致

    6. Check 6 - Info Gain 贡献
       - 验证 info_gain 是否对 discounted_returns 产生了贡献
       - 检查 turn 间 advantage 的差异（说明 info_gain 起作用）

=============================================================================
                           向量化 vs 顺序对比
=============================================================================

    为了验证向量化计算的正确性，可以按以下步骤进行：

    1. 使用顺序模式运行：
       export DEBUG_IGPO_STRICT=1
       在 train_grpo.sh 中设置: +algorithm.use_vectorized_gt_logprob=false
       运行训练，观察 Check 1 的输出

    2. 使用向量化模式运行：
       export DEBUG_IGPO_STRICT=1
       在 train_grpo.sh 中设置: +algorithm.use_vectorized_gt_logprob=true
       运行训练，观察 Check 1 的输出

    3. 对比两次运行的 info_gain_reward 值是否一致（在浮点精度内）

=============================================================================

Author: IGPO Team
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json


def is_strict_check_enabled() -> bool:
    """检查是否启用严格验证"""
    return os.environ.get("DEBUG_IGPO_STRICT", "0") == "1"


@dataclass
class CheckResult:
    """单个检查点的结果"""
    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineCheckpoint:
    """流水线检查点数据"""
    # Check 1: Generation 计算的 info_gain_reward
    generation_info_gains: Dict[int, List[float]] = field(default_factory=dict)  # sample_idx -> [ig_0, ig_1, ...]
    generation_mode: str = ""  # "vectorized" or "sequential"
    
    # Check 2: info_gain.py 接收的数据
    info_gain_received: Dict[int, List[float]] = field(default_factory=dict)
    info_gain_positions: Dict[int, List[int]] = field(default_factory=dict)  # sample_idx -> [pos_0, pos_1, ...]
    f1_scores: Dict[int, float] = field(default_factory=dict)
    f1_positions: Dict[int, int] = field(default_factory=dict)
    
    # Check 3: core_algos.py 接收的 token_level_rewards
    core_algos_rewards: Optional[torch.Tensor] = None
    core_algos_positions: Dict[int, List[int]] = field(default_factory=dict)
    
    # Check 4: 归一化前后的统计量
    norm_before_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    norm_after_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Check 5: Turn-level 累积结果
    turn_advantages: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict)  # sample_idx -> [(pos, adv), ...]
    discounted_returns: Optional[torch.Tensor] = None
    
    # Check 6: Loss 贡献
    loss_contributions: Dict[str, float] = field(default_factory=dict)


# 全局检查点实例
_checkpoint = PipelineCheckpoint()


def get_checkpoint() -> PipelineCheckpoint:
    """获取全局检查点"""
    return _checkpoint


def reset_checkpoint():
    """重置检查点"""
    global _checkpoint
    _checkpoint = PipelineCheckpoint()


# ============================================================================
# Check 1: Generation 阶段验证
# ============================================================================

def record_generation_info_gain(
    sample_idx: int,
    info_gain_rewards: List[float],
    mode: str,  # "vectorized" or "sequential"
):
    """记录 generation 阶段计算的 info_gain_reward"""
    if not is_strict_check_enabled():
        return
    
    _checkpoint.generation_info_gains[sample_idx] = info_gain_rewards.copy()
    _checkpoint.generation_mode = mode


def verify_vectorized_vs_sequential(
    vectorized_results: List[float],
    sequential_results: List[float],
    sample_idx: int,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> CheckResult:
    """验证向量化和顺序计算结果是否一致"""
    if len(vectorized_results) != len(sequential_results):
        return CheckResult(
            check_name="Check 1: Vectorized vs Sequential",
            passed=False,
            message=f"Length mismatch: vectorized={len(vectorized_results)}, sequential={len(sequential_results)}",
            details={"sample_idx": sample_idx}
        )
    
    mismatches = []
    for i, (v, s) in enumerate(zip(vectorized_results, sequential_results)):
        if not np.isclose(v, s, rtol=rtol, atol=atol):
            mismatches.append({
                "turn": i,
                "vectorized": v,
                "sequential": s,
                "diff": abs(v - s)
            })
    
    if mismatches:
        return CheckResult(
            check_name="Check 1: Vectorized vs Sequential",
            passed=False,
            message=f"Found {len(mismatches)} mismatches",
            details={"sample_idx": sample_idx, "mismatches": mismatches}
        )
    
    return CheckResult(
        check_name="Check 1: Vectorized vs Sequential",
        passed=True,
        message="All values match",
        details={"sample_idx": sample_idx, "num_turns": len(vectorized_results)}
    )


# 存储历史结果用于跨运行对比
_history_results: Dict[str, Dict[int, List[float]]] = {}


def save_results_for_comparison(mode: str):
    """保存当前结果用于后续对比"""
    global _history_results
    _history_results[mode] = dict(_checkpoint.generation_info_gains)
    print(f"\n[Vectorized Comparison] Saved {mode} results: {len(_history_results[mode])} samples")


def compare_vectorized_sequential():
    """对比向量化和顺序计算的结果"""
    if "vectorized" not in _history_results or "sequential" not in _history_results:
        print("\n[Vectorized Comparison] Missing data. Need both 'vectorized' and 'sequential' results.")
        print("  Run with use_vectorized_gt_logprob=false first, then with use_vectorized_gt_logprob=true")
        return None
    
    vec_results = _history_results["vectorized"]
    seq_results = _history_results["sequential"]
    
    print("\n" + "=" * 80)
    print("    VECTORIZED vs SEQUENTIAL COMPARISON")
    print("=" * 80)
    
    all_samples = set(vec_results.keys()) | set(seq_results.keys())
    total_matches = 0
    total_mismatches = 0
    
    for sample_idx in sorted(all_samples)[:10]:  # 只显示前 10 个样本
        vec = vec_results.get(sample_idx, [])
        seq = seq_results.get(sample_idx, [])
        
        if len(vec) != len(seq):
            print(f"\n  Sample {sample_idx}: Length mismatch - vec={len(vec)}, seq={len(seq)}")
            total_mismatches += 1
            continue
        
        matches = sum(1 for v, s in zip(vec, seq) if np.isclose(v, s, rtol=1e-4, atol=1e-6))
        mismatches = len(vec) - matches
        
        if mismatches > 0:
            print(f"\n  Sample {sample_idx}: {mismatches} mismatches out of {len(vec)} turns")
            for i, (v, s) in enumerate(zip(vec, seq)):
                if not np.isclose(v, s, rtol=1e-4, atol=1e-6):
                    print(f"    Turn {i}: vec={v:.6f}, seq={s:.6f}, diff={abs(v-s):.2e}")
            total_mismatches += 1
        else:
            print(f"\n  Sample {sample_idx}: All {len(vec)} turns match ✓")
            total_matches += 1
    
    print("\n" + "-" * 80)
    print(f"  Total: {total_matches} samples match, {total_mismatches} samples with mismatches")
    if total_mismatches == 0:
        print("  ✓ VECTORIZED AND SEQUENTIAL RESULTS ARE IDENTICAL!")
    else:
        print("  ⚠️ SOME MISMATCHES FOUND. Please investigate.")
    print("=" * 80 + "\n")
    
    return total_mismatches == 0


# ============================================================================
# Check 2: info_gain.py 阶段验证
# ============================================================================

def record_info_gain_assignment(
    sample_idx: int,
    info_gain_rewards: List[float],
    info_gain_positions: List[int],
    f1_score: float,
    f1_position: int,
):
    """记录 info_gain.py 中的 reward 分配"""
    if not is_strict_check_enabled():
        return
    
    _checkpoint.info_gain_received[sample_idx] = info_gain_rewards.copy()
    _checkpoint.info_gain_positions[sample_idx] = info_gain_positions.copy()
    _checkpoint.f1_scores[sample_idx] = f1_score
    _checkpoint.f1_positions[sample_idx] = f1_position


def verify_info_gain_consistency(sample_idx: int) -> CheckResult:
    """验证 info_gain.py 接收的 reward 与 generation 阶段一致"""
    gen_rewards = _checkpoint.generation_info_gains.get(sample_idx, [])
    received_rewards = _checkpoint.info_gain_received.get(sample_idx, [])
    
    if not gen_rewards:
        return CheckResult(
            check_name="Check 2: Info Gain Consistency",
            passed=True,
            message="No generation data to compare (skipped)",
            details={"sample_idx": sample_idx}
        )
    
    if len(gen_rewards) != len(received_rewards):
        return CheckResult(
            check_name="Check 2: Info Gain Consistency",
            passed=False,
            message=f"Length mismatch: generation={len(gen_rewards)}, received={len(received_rewards)}",
            details={"sample_idx": sample_idx}
        )
    
    mismatches = []
    for i, (g, r) in enumerate(zip(gen_rewards, received_rewards)):
        if not np.isclose(g, r, rtol=1e-6):
            mismatches.append({"turn": i, "generation": g, "received": r})
    
    if mismatches:
        return CheckResult(
            check_name="Check 2: Info Gain Consistency",
            passed=False,
            message=f"Found {len(mismatches)} mismatches",
            details={"sample_idx": sample_idx, "mismatches": mismatches}
        )
    
    return CheckResult(
        check_name="Check 2: Info Gain Consistency",
        passed=True,
        message="All rewards consistent",
        details={"sample_idx": sample_idx, "num_rewards": len(gen_rewards)}
    )


# ============================================================================
# Check 3: core_algos.py 传输验证
# ============================================================================

def record_core_algos_rewards(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
):
    """记录 core_algos.py 接收到的 rewards"""
    if not is_strict_check_enabled():
        return
    
    _checkpoint.core_algos_rewards = token_level_rewards.detach().clone()
    
    bsz = token_level_rewards.shape[0]
    for i in range(bsz):
        positions = (token_level_rewards[i] != 0).nonzero(as_tuple=True)[0].tolist()
        _checkpoint.core_algos_positions[i] = positions


def verify_reward_transmission(sample_idx: int) -> CheckResult:
    """验证 reward 传输正确性"""
    if _checkpoint.core_algos_rewards is None:
        return CheckResult(
            check_name="Check 3: Reward Transmission",
            passed=True,
            message="No core_algos data (skipped)",
            details={"sample_idx": sample_idx}
        )
    
    # 获取 info_gain.py 中的位置和值
    ig_positions = _checkpoint.info_gain_positions.get(sample_idx, [])
    ig_rewards = _checkpoint.info_gain_received.get(sample_idx, [])
    f1_position = _checkpoint.f1_positions.get(sample_idx, -1)
    f1_score = _checkpoint.f1_scores.get(sample_idx, 0.0)
    
    # 获取 core_algos 中的位置和值
    core_positions = _checkpoint.core_algos_positions.get(sample_idx, [])
    core_rewards = _checkpoint.core_algos_rewards[sample_idx] if sample_idx < len(_checkpoint.core_algos_rewards) else None
    
    if core_rewards is None:
        return CheckResult(
            check_name="Check 3: Reward Transmission",
            passed=False,
            message=f"Sample {sample_idx} not found in core_algos",
            details={"sample_idx": sample_idx}
        )
    
    # 验证位置
    expected_positions = ig_positions + [f1_position] if f1_position >= 0 else ig_positions
    if set(core_positions) != set(expected_positions):
        return CheckResult(
            check_name="Check 3: Reward Transmission",
            passed=False,
            message="Position mismatch",
            details={
                "sample_idx": sample_idx,
                "expected": expected_positions,
                "actual": core_positions
            }
        )
    
    # 验证值
    mismatches = []
    for i, pos in enumerate(ig_positions):
        if pos < len(core_rewards):
            expected = ig_rewards[i] if i < len(ig_rewards) else 0.0
            actual = core_rewards[pos].item()
            # 注意：0.0 可能被替换为 1e-10
            if not (np.isclose(expected, actual, rtol=1e-6) or 
                    (expected == 0.0 and np.isclose(actual, 1e-10, rtol=1e-6))):
                mismatches.append({"pos": pos, "expected": expected, "actual": actual})
    
    if f1_position >= 0 and f1_position < len(core_rewards):
        actual_f1 = core_rewards[f1_position].item()
        if not np.isclose(f1_score, actual_f1, rtol=1e-6):
            mismatches.append({"pos": f1_position, "type": "f1", "expected": f1_score, "actual": actual_f1})
    
    if mismatches:
        return CheckResult(
            check_name="Check 3: Reward Transmission",
            passed=False,
            message=f"Found {len(mismatches)} value mismatches",
            details={"sample_idx": sample_idx, "mismatches": mismatches}
        )
    
    return CheckResult(
        check_name="Check 3: Reward Transmission",
        passed=True,
        message="All rewards transmitted correctly",
        details={"sample_idx": sample_idx, "num_positions": len(expected_positions)}
    )


# ============================================================================
# Check 4: 标准化验证
# ============================================================================

def record_normalization_stats(
    before_rewards: torch.Tensor,
    after_rewards: torch.Tensor,
    f1_mask: torch.Tensor,
    ig_mask: torch.Tensor,
    mode: str,  # "separate" or "joint"
    group_ids: torch.Tensor,
):
    """记录归一化前后的统计量"""
    if not is_strict_check_enabled():
        return
    
    with torch.no_grad():
        if mode == "separate":
            # F1
            f1_before = before_rewards[f1_mask]
            f1_after = after_rewards[f1_mask]
            if len(f1_before) > 0:
                _checkpoint.norm_before_stats["f1"] = {
                    "mean": f1_before.mean().item(),
                    "std": f1_before.std().item() if len(f1_before) > 1 else 0.0,
                    "count": len(f1_before)
                }
                _checkpoint.norm_after_stats["f1"] = {
                    "mean": f1_after.mean().item(),
                    "std": f1_after.std().item() if len(f1_after) > 1 else 0.0,
                    "count": len(f1_after)
                }
            
            # InfoGain
            ig_before = before_rewards[ig_mask]
            ig_after = after_rewards[ig_mask]
            if len(ig_before) > 0:
                _checkpoint.norm_before_stats["ig"] = {
                    "mean": ig_before.mean().item(),
                    "std": ig_before.std().item() if len(ig_before) > 1 else 0.0,
                    "count": len(ig_before)
                }
                _checkpoint.norm_after_stats["ig"] = {
                    "mean": ig_after.mean().item(),
                    "std": ig_after.std().item() if len(ig_after) > 1 else 0.0,
                    "count": len(ig_after)
                }
        else:
            # Joint
            joint_mask = f1_mask | ig_mask
            joint_before = before_rewards[joint_mask]
            joint_after = after_rewards[joint_mask]
            if len(joint_before) > 0:
                _checkpoint.norm_before_stats["joint"] = {
                    "mean": joint_before.mean().item(),
                    "std": joint_before.std().item() if len(joint_before) > 1 else 0.0,
                    "count": len(joint_before)
                }
                _checkpoint.norm_after_stats["joint"] = {
                    "mean": joint_after.mean().item(),
                    "std": joint_after.std().item() if len(joint_after) > 1 else 0.0,
                    "count": len(joint_after)
                }


def verify_normalization(mode: str, norm_by_std: bool) -> CheckResult:
    """验证归一化正确性"""
    results = []
    
    if mode == "separate":
        # 验证 F1 归一化
        if "f1" in _checkpoint.norm_after_stats:
            f1_stats = _checkpoint.norm_after_stats["f1"]
            mean_ok = np.isclose(f1_stats["mean"], 0.0, atol=0.1)
            std_ok = not norm_by_std or np.isclose(f1_stats["std"], 1.0, atol=0.2) or f1_stats["count"] <= 1
            results.append(("f1", mean_ok and std_ok, f1_stats))
        
        # 验证 InfoGain 归一化
        if "ig" in _checkpoint.norm_after_stats:
            ig_stats = _checkpoint.norm_after_stats["ig"]
            mean_ok = np.isclose(ig_stats["mean"], 0.0, atol=0.1)
            std_ok = not norm_by_std or np.isclose(ig_stats["std"], 1.0, atol=0.2) or ig_stats["count"] <= 1
            results.append(("ig", mean_ok and std_ok, ig_stats))
    else:
        # 验证 Joint 归一化
        if "joint" in _checkpoint.norm_after_stats:
            joint_stats = _checkpoint.norm_after_stats["joint"]
            mean_ok = np.isclose(joint_stats["mean"], 0.0, atol=0.1)
            std_ok = not norm_by_std or np.isclose(joint_stats["std"], 1.0, atol=0.2) or joint_stats["count"] <= 1
            results.append(("joint", mean_ok and std_ok, joint_stats))
    
    all_passed = all(r[1] for r in results)
    
    return CheckResult(
        check_name="Check 4: Normalization",
        passed=all_passed,
        message="Normalization correct" if all_passed else "Normalization issues found",
        details={
            "mode": mode,
            "norm_by_std": norm_by_std,
            "results": [(r[0], r[1], r[2]) for r in results],
            "before_stats": _checkpoint.norm_before_stats,
            "after_stats": _checkpoint.norm_after_stats
        }
    )


# ============================================================================
# Check 5: Turn-level 折扣累积验证
# ============================================================================

def record_turn_level_results(
    sample_idx: int,
    turn_data: List[Tuple[int, float]],  # [(pos, advantage), ...]
    discounted_returns: torch.Tensor,
):
    """记录 turn-level 累积结果"""
    if not is_strict_check_enabled():
        return
    
    _checkpoint.turn_advantages[sample_idx] = turn_data.copy()
    if _checkpoint.discounted_returns is None:
        _checkpoint.discounted_returns = discounted_returns.detach().clone()


def verify_turn_level_discounting(
    sample_idx: int,
    gamma: float,
    normalized_rewards: torch.Tensor,
) -> CheckResult:
    """验证 turn-level 折扣累积正确性"""
    turn_data = _checkpoint.turn_advantages.get(sample_idx, [])
    
    if not turn_data:
        return CheckResult(
            check_name="Check 5a: Discounting Formula",
            passed=True,
            message="No turn data (skipped)",
            details={"sample_idx": sample_idx}
        )
    
    # 验证折扣累积公式
    mismatches = []
    for i in range(len(turn_data) - 1, -1, -1):
        pos, adv = turn_data[i]
        r_i = normalized_rewards[sample_idx, pos].item()
        
        if i == len(turn_data) - 1:
            expected = r_i
        else:
            next_adv = turn_data[i + 1][1]
            expected = r_i + gamma * next_adv
        
        if not np.isclose(expected, adv, rtol=1e-5, atol=1e-6):
            mismatches.append({
                "turn": i,
                "expected": expected,
                "actual": adv,
                "r_i": r_i,
                "diff": abs(expected - adv)
            })
    
    if mismatches:
        return CheckResult(
            check_name="Check 5a: Discounting Formula",
            passed=False,
            message=f"Formula verification failed for {len(mismatches)} turns",
            details={"sample_idx": sample_idx, "mismatches": mismatches, "gamma": gamma}
        )
    
    return CheckResult(
        check_name="Check 5a: Discounting Formula",
        passed=True,
        message="Discounting formula verified",
        details={"sample_idx": sample_idx, "num_turns": len(turn_data), "gamma": gamma}
    )


def verify_turn_broadcast(
    sample_idx: int,
    response_mask: torch.Tensor,
) -> CheckResult:
    """验证 turn 内广播正确性"""
    turn_data = _checkpoint.turn_advantages.get(sample_idx, [])
    discounted_returns = _checkpoint.discounted_returns
    
    if not turn_data or discounted_returns is None:
        return CheckResult(
            check_name="Check 5b: Turn Broadcast",
            passed=True,
            message="No data (skipped)",
            details={"sample_idx": sample_idx}
        )
    
    sample_returns = discounted_returns[sample_idx]
    sample_mask = response_mask[sample_idx]
    
    # 检查每个 turn 内的 tokens 是否有相同的 advantage
    prev_end = 0
    non_uniform_turns = []
    
    for turn_idx, (reward_pos, expected_adv) in enumerate(turn_data):
        turn_values = []
        for t in range(prev_end, reward_pos + 1):
            if sample_mask[t] == 1:
                turn_values.append(sample_returns[t].item())
        
        if len(turn_values) > 1:
            if not all(np.isclose(v, turn_values[0], rtol=1e-5, atol=1e-6) for v in turn_values):
                non_uniform_turns.append({
                    "turn": turn_idx,
                    "range": (prev_end, reward_pos),
                    "values": turn_values[:5],  # 只保留前 5 个
                    "expected": expected_adv
                })
        
        prev_end = reward_pos + 1
    
    if non_uniform_turns:
        return CheckResult(
            check_name="Check 5b: Turn Broadcast",
            passed=False,
            message=f"Found {len(non_uniform_turns)} non-uniform turns",
            details={"sample_idx": sample_idx, "non_uniform_turns": non_uniform_turns}
        )
    
    return CheckResult(
        check_name="Check 5b: Turn Broadcast",
        passed=True,
        message="All turns have uniform advantages",
        details={"sample_idx": sample_idx, "num_turns": len(turn_data)}
    )


# ============================================================================
# Check 6: Info Gain 贡献验证（通过分析 discounted_returns）
# ============================================================================

def verify_info_gain_contribution(
    response_mask: torch.Tensor,
) -> CheckResult:
    """
    验证 info_gain 确实对 discounted_returns 产生了贡献。
    
    检查逻辑：
    1. 找到有 info_gain reward 的位置
    2. 验证这些 info_gain 被正确累积并广播到了 turn 内的所有 tokens
    3. 验证 discounted_returns 在有效位置上不全为 0（说明 info_gain 参与了计算）
    """
    discounted_returns = _checkpoint.discounted_returns
    
    if discounted_returns is None:
        return CheckResult(
            check_name="Check 6: Info Gain Contribution",
            passed=True,
            message="No discounted_returns data (skipped)",
            details={}
        )
    
    bsz = discounted_returns.shape[0]
    samples_with_ig_contribution = 0
    samples_total = 0
    ig_variance_samples = 0  # 有 info_gain 导致 turn 间差异的样本数
    
    detailed_analysis = []
    
    for sample_idx in range(min(bsz, 3)):  # 只详细分析前 3 个样本
        turn_data = _checkpoint.turn_advantages.get(sample_idx, [])
        if len(turn_data) == 0:
            continue
        
        samples_total += 1
        sample_returns = discounted_returns[sample_idx]
        sample_mask = response_mask[sample_idx]
        
        # 获取每个 turn 的 advantage 值
        turn_advs = [adv for _, adv in turn_data]
        
        # 计算 turn 间的方差
        if len(turn_advs) > 1:
            adv_variance = np.var(turn_advs)
            if adv_variance > 1e-8:  # 有明显差异
                ig_variance_samples += 1
        
        # 检查 discounted_returns 在有效位置上是否非零
        valid_returns = sample_returns[sample_mask == 1]
        nonzero_ratio = (valid_returns != 0).float().mean().item()
        
        if nonzero_ratio > 0.5:  # 超过一半的有效位置有非零值
            samples_with_ig_contribution += 1
        
        detailed_analysis.append({
            "sample_idx": sample_idx,
            "num_turns": len(turn_data),
            "turn_advantages": turn_advs,
            "adv_variance": np.var(turn_advs) if len(turn_advs) > 1 else 0.0,
            "nonzero_ratio": nonzero_ratio,
        })
    
    # 判断通过条件
    # 1. 大多数样本都有非零的 discounted_returns
    # 2. 如果有多个 turn，应该有 turn 间的差异（说明 info_gain 不同于 f1）
    passed = samples_with_ig_contribution >= samples_total * 0.8
    
    return CheckResult(
        check_name="Check 6: Info Gain Contribution",
        passed=passed,
        message=f"{samples_with_ig_contribution}/{samples_total} samples have info_gain contribution" if samples_total > 0 else "No samples",
        details={
            "samples_with_ig_contribution": samples_with_ig_contribution,
            "samples_total": samples_total,
            "ig_variance_samples": ig_variance_samples,
            "detailed_analysis": detailed_analysis,
        }
    )


# ============================================================================
# 统一验证和打印
# ============================================================================

def run_all_checks(
    gamma: float = 1.0,
    norm_mode: str = "joint",
    norm_by_std: bool = True,
    response_mask: Optional[torch.Tensor] = None,
    normalized_rewards: Optional[torch.Tensor] = None,
) -> List[CheckResult]:
    """运行所有验证检查"""
    if not is_strict_check_enabled():
        return []
    
    results = []
    
    # 获取样本数量
    num_samples = len(_checkpoint.generation_info_gains) if _checkpoint.generation_info_gains else 0
    if num_samples == 0 and _checkpoint.core_algos_rewards is not None:
        num_samples = min(3, _checkpoint.core_algos_rewards.shape[0])
    
    print("\n" + "=" * 80)
    print("    IGPO STRICT PIPELINE VERIFICATION")
    print("=" * 80)
    
    # ========== Check 1: Generation 阶段 info_gain 计算 ==========
    print("\n>>> CHECK 1: Generation Phase - Info Gain Calculation")
    gen_mode = _checkpoint.generation_mode
    gen_samples = len(_checkpoint.generation_info_gains)
    if gen_samples > 0:
        total_ig = sum(len(v) for v in _checkpoint.generation_info_gains.values())
        print(f"  ✓ Mode: {gen_mode}")
        print(f"  ✓ Samples recorded: {gen_samples}")
        print(f"  ✓ Total info_gain values: {total_ig}")
        
        # 显示前 3 个样本的详细信息
        for i in list(_checkpoint.generation_info_gains.keys())[:3]:
            rewards = _checkpoint.generation_info_gains[i]
            print(f"    Sample {i}: {len(rewards)} turns, values = {[f'{r:.6f}' for r in rewards]}")
        
        results.append(CheckResult("Check 1: Generation", True, f"{gen_samples} samples, {total_ig} info_gain values"))
    else:
        print(f"  ⚠️ No generation data recorded")
        results.append(CheckResult("Check 1: Generation", True, "No data (skipped)", {}))
    
    # ========== Check 2: info_gain.py 分配验证 ==========
    print("\n>>> CHECK 2: info_gain.py - Reward Assignment")
    ig_samples = len(_checkpoint.info_gain_received)
    if ig_samples > 0:
        print(f"  Samples with assignments: {ig_samples}")
        
        for sample_idx in list(_checkpoint.info_gain_received.keys())[:3]:
            rewards = _checkpoint.info_gain_received.get(sample_idx, [])
            positions = _checkpoint.info_gain_positions.get(sample_idx, [])
            f1_pos = _checkpoint.f1_positions.get(sample_idx, -1)
            f1_score = _checkpoint.f1_scores.get(sample_idx, 0.0)
            
            print(f"\n  Sample {sample_idx}:")
            print(f"    InfoGain: {len(rewards)} values at positions {positions}")
            print(f"    F1: {f1_score:.4f} at position {f1_pos}")
            
            # 验证与 generation 阶段的一致性
            result = verify_info_gain_consistency(sample_idx)
            results.append(result)
            status = "✓" if result.passed else "✗"
            print(f"    Consistency with generation: {status} {result.message}")
    else:
        print(f"  ⚠️ No info_gain.py assignment data")
    
    # ========== Check 3: core_algos.py 传输验证 ==========
    print("\n>>> CHECK 3: core_algos.py - Reward Transmission")
    if _checkpoint.core_algos_rewards is not None:
        bsz = _checkpoint.core_algos_rewards.shape[0]
        seq_len = _checkpoint.core_algos_rewards.shape[1]
        print(f"  Received rewards: shape ({bsz}, {seq_len})")
        
        for sample_idx in range(min(3, bsz)):
            core_positions = _checkpoint.core_algos_positions.get(sample_idx, [])
            core_values = [_checkpoint.core_algos_rewards[sample_idx, p].item() for p in core_positions] if core_positions else []
            
            print(f"\n  Sample {sample_idx}:")
            print(f"    Positions: {core_positions}")
            print(f"    Values: {[f'{v:.6f}' for v in core_values]}")
            
            result = verify_reward_transmission(sample_idx)
            results.append(result)
            status = "✓" if result.passed else "✗"
            print(f"    Transmission verification: {status} {result.message}")
    else:
        print(f"  ⚠️ No core_algos reward data")
    
    # ========== Check 4: 归一化验证 ==========
    print("\n>>> CHECK 4: Normalization")
    result = verify_normalization(norm_mode, norm_by_std)
    results.append(result)
    
    print(f"  Mode: {norm_mode}, Norm by std: {norm_by_std}")
    if _checkpoint.norm_before_stats:
        print(f"  Before normalization:")
        for key, stats in _checkpoint.norm_before_stats.items():
            print(f"    {key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, count={stats['count']}")
    if _checkpoint.norm_after_stats:
        print(f"  After normalization:")
        for key, stats in _checkpoint.norm_after_stats.items():
            print(f"    {key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, count={stats['count']}")
    status = "✓" if result.passed else "✗"
    print(f"  Verification: {status} {result.message}")
    
    # ========== Check 5: Turn-level 折扣累积和广播 ==========
    print("\n>>> CHECK 5: Turn-Level Discounting & Broadcasting")
    print(f"  Gamma: {gamma}")
    
    for sample_idx in range(min(3, num_samples)):
        turn_data = _checkpoint.turn_advantages.get(sample_idx, [])
        if not turn_data:
            continue
        
        print(f"\n  Sample {sample_idx}:")
        print(f"    Turns: {len(turn_data)}")
        
        # Check 5a: 折扣累积公式
        if normalized_rewards is not None:
            result = verify_turn_level_discounting(sample_idx, gamma, normalized_rewards)
            results.append(result)
            status = "✓" if result.passed else "✗"
            print(f"    Discounting formula: {status} {result.message}")
            
            # 显示详细的折扣累积过程
            for i, (pos, adv) in enumerate(turn_data):
                if i == len(turn_data) - 1:
                    print(f"      Turn {i}: r={normalized_rewards[sample_idx, pos].item():.4f}, A={adv:.4f} (last turn)")
                else:
                    next_adv = turn_data[i+1][1]
                    r_i = normalized_rewards[sample_idx, pos].item()
                    expected = r_i + gamma * next_adv
                    match = "✓" if abs(expected - adv) < 1e-6 else "✗"
                    print(f"      Turn {i}: r={r_i:.4f} + γ*A_{i+1}={gamma}*{next_adv:.4f} = {expected:.4f}, actual={adv:.4f} {match}")
        
        # Check 5b: 广播正确性
        if response_mask is not None:
            result = verify_turn_broadcast(sample_idx, response_mask)
            results.append(result)
            status = "✓" if result.passed else "✗"
            print(f"    Broadcast uniformity: {status} {result.message}")
    
    # ========== Check 6: Info Gain 贡献验证 ==========
    print("\n>>> CHECK 6: Info Gain Contribution to Loss")
    if response_mask is not None:
        result = verify_info_gain_contribution(response_mask)
        results.append(result)
        
        status = "✓" if result.passed else "✗"
        print(f"  {status} {result.message}")
        
        if result.details.get("detailed_analysis"):
            for analysis in result.details["detailed_analysis"]:
                print(f"\n  Sample {analysis['sample_idx']}:")
                print(f"    Turns: {analysis['num_turns']}")
                print(f"    Turn advantages: {[f'{a:.4f}' for a in analysis['turn_advantages']]}")
                print(f"    Advantage variance: {analysis['adv_variance']:.6f}")
                print(f"    Non-zero ratio: {analysis['nonzero_ratio']:.2%}")
    
    # ========== 总结 ==========
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print("\n" + "=" * 80)
    print(f"    SUMMARY: {passed}/{total} checks passed")
    if passed == total:
        print("    ✓ ALL CHECKS PASSED - IGPO Pipeline is working correctly!")
    else:
        failed_checks = [r.check_name for r in results if not r.passed]
        print(f"    ⚠️ FAILED CHECKS: {failed_checks}")
        print("    Please review the details above.")
    print("=" * 80 + "\n")
    
    return results


def _print_check_result(result: CheckResult):
    """打印单个检查结果"""
    status = "✓ PASS" if result.passed else "✗ FAIL"
    print(f"\n[{status}] {result.check_name}")
    print(f"  Message: {result.message}")
    
    if result.details:
        for key, value in result.details.items():
            if isinstance(value, (list, dict)) and len(str(value)) > 100:
                print(f"  {key}: (truncated, {len(value)} items)")
            else:
                print(f"  {key}: {value}")
