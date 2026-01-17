#!/bin/bash

# Get the directory where this shell script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# ============================================================
# 配置区域 - 可以修改这些变量或通过命令行参数覆盖
# ============================================================

# 默认的评估日志目录 (可通过第一个命令行参数或环境变量覆盖)
DEFAULT_BASE_DIR="${SCRIPT_DIR}/eval_log"

# 默认的输出目录 (可通过第二个命令行参数覆盖，留空则使用 base_dir/figures)
DEFAULT_OUTPUT_DIR=""

# ============================================================
# 参数解析
# ============================================================

# 优先级: 命令行参数 > 环境变量 > 默认值
BASE_DIR="${1:-${EVAL_LOG_DIR:-$DEFAULT_BASE_DIR}}"
OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"

# 显示帮助信息
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "用法: $0 [BASE_DIR] [OUTPUT_DIR]"
    echo ""
    echo "参数:"
    echo "  BASE_DIR    包含 metric_step_*.json 文件的目录"
    echo "              默认: \$EVAL_LOG_DIR 或 ${DEFAULT_BASE_DIR}"
    echo "  OUTPUT_DIR  保存图片的输出目录"
    echo "              默认: BASE_DIR/figures"
    echo ""
    echo "示例:"
    echo "  $0                                    # 使用默认路径"
    echo "  $0 ./my_eval_log                      # 指定输入目录"
    echo "  $0 ./my_eval_log ./my_figures         # 指定输入和输出目录"
    echo "  EVAL_LOG_DIR=./logs $0                # 使用环境变量"
    exit 0
fi

# ============================================================
# 执行可视化
# ============================================================

echo "=========================================="
echo "Metrics Visualization"
echo "=========================================="
echo "输入目录: ${BASE_DIR}"
if [[ -n "$OUTPUT_DIR" ]]; then
    echo "输出目录: ${OUTPUT_DIR}"
else
    echo "输出目录: ${BASE_DIR}/figures (默认)"
fi
echo "=========================================="

# 检查输入目录是否存在
if [[ ! -d "$BASE_DIR" ]]; then
    echo "错误: 目录不存在: ${BASE_DIR}"
    exit 1
fi

# 构建命令
CMD="python3 ${SCRIPT_DIR}/metrics_visualization.py --base_dir ${BASE_DIR}"

if [[ -n "$OUTPUT_DIR" ]]; then
    CMD="${CMD} --output_dir ${OUTPUT_DIR}"
fi

# 执行
echo "执行命令: ${CMD}"
echo ""

$CMD
exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    echo ""
    echo "可视化完成!"
else
    echo ""
    echo "可视化失败，退出码: ${exit_code}"
fi

exit $exit_code
