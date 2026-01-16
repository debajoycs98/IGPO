# Get the directory where this shell script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

base_dir="/ossfs/workspace/linyang/FactAgent/DeepResearcher/eval_log_3B_ours"

python3 metrics_visualization.py --base_dir $base_dir 
exit $exit_code 