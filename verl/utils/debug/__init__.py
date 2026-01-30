# Original debug utilities (must be first to maintain compatibility)
from .performance import log_gpu_memory_usage, GPUMemoryLogger

# IGPO Debug Utilities
from .igpo_pipeline_checker import (
    is_strict_check_enabled,
    get_checkpoint,
    reset_checkpoint,
    record_generation_info_gain,
    record_info_gain_assignment,
    record_core_algos_rewards,
    record_normalization_stats,
    record_turn_level_results,
    verify_vectorized_vs_sequential,
    verify_info_gain_contribution,
    run_all_checks,
    save_results_for_comparison,
    compare_vectorized_sequential,
)

# IGPO Full Checker
from .igpo_full_checker import (
    is_full_check_enabled,
    get_full_checker,
    reset_full_checker,
    IGPOFullChecker,
)

# IGPO Unit Tests
from .igpo_unit_tests import run_unit_tests, IGPOUnitTester
