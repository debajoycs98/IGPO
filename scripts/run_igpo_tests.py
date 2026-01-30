#!/usr/bin/env python3
# Copyright 2025 IGPO Team
# IGPO 完整测试套件

"""
运行方式:
    python scripts/run_igpo_tests.py [--unit] [--full]
    
    --unit: 只运行单元测试
    --full: 只运行完整验证（需要在训练中启用）
    无参数: 运行单元测试
"""

import sys
import os

# 添加项目根目录到 path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def run_unit_tests():
    """运行单元测试"""
    print("\n" + "="*70)
    print(" Running IGPO Unit Tests")
    print("="*70 + "\n")
    
    # 直接导入单元测试模块，避免导入整个 verl 包
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "igpo_unit_tests",
        os.path.join(project_root, "verl/utils/debug/igpo_unit_tests.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module.run_unit_tests()


def print_full_check_instructions():
    """打印完整验证说明"""
    print("\n" + "="*70)
    print(" IGPO Full Check Instructions")
    print("="*70)
    print("""
完整验证需要在训练过程中运行，请按以下步骤操作：

1. 在 train_grpo.sh 中添加环境变量:
   export DEBUG_IGPO_FULL_CHECK=1

2. 运行训练:
   bash train_grpo.sh

3. 观察终端输出，验证结果会自动打印

验证覆盖的检查点:
  A: GT LogProb 计算值
  B: Info Gain 计算 (prev - curr)
  C: Info Gain 一致性 (generation → info_gain.py)
  D: 轮次位置分配
  E: Token 位置分配
  F: 奖励传输 (info_gain.py → core_algos.py)
  G: 标准化验证
  H: Turn-level 折扣累积
  I: Turn 内广播
  J: Response Mask
  K: 向量化 vs 顺序一致性
  L: Policy Loss 计算
  M: 梯度验证
  N: Info Gain 梯度贡献
""")


def main():
    args = sys.argv[1:]
    
    if not args or "--unit" in args:
        success = run_unit_tests()
        if not success:
            print("\n❌ Unit tests FAILED")
            sys.exit(1)
        else:
            print("\n✓ Unit tests PASSED")
    
    if "--full" in args:
        print_full_check_instructions()
    
    if not args:
        print("\n" + "-"*70)
        print("提示: 运行 'python scripts/run_igpo_tests.py --full' 查看完整验证说明")
        print("-"*70)


if __name__ == "__main__":
    main()
