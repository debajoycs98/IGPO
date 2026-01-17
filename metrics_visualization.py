import json
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import argparse
import pandas as pd


class MetricsVisualizer:
    """
    Loads metrics data from a JSONL file associated with a project/experiment
    and provides methods for visualizing various metrics.
    """
    def __init__(self, base_dir, output_dir=None):
        """
        Args:
            base_dir: Directory containing metric_step_*.json files
            output_dir: Directory to save output figures. If None, defaults to base_dir/figures
        """
        self.base_dir = base_dir
        self.output_dir = output_dir if output_dir else os.path.join(base_dir, 'figures')

        if not os.path.exists(base_dir):
            print(f"错误: 目录不存在: {base_dir}")
            self.dataframe = pd.DataFrame()
            return

        file_path_list = os.listdir(base_dir)

        metric_files = [f for f in file_path_list if f.startswith('metric_step_') and f.endswith('.json')]
        
        if not metric_files:
            print(f"警告: 在 {base_dir} 中没有找到 metric_step_*.json 文件")
            self.dataframe = pd.DataFrame()
            return
            
        try:
            metric_files.sort(key=lambda f: int(f.replace('metric_step_', '').replace('.json', '')))
        except ValueError:
            print("Error: Could not parse step numbers from filenames. Please ensure they follow the 'metric_step_x.json' format.")
            self.dataframe = pd.DataFrame()
            return
            
        datas = []
        for file_name in metric_files:
            file_path = os.path.join(base_dir, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f: 
                    if 'metric' in file_name:
                        data = json.load(f)
                        datas.append(data)
                        print(f"成功加载: {file_name}")
            except json.JSONDecodeError:
                print(f"错误: 文件 '{file_name}' 不是有效的 JSON 格式。")
            except IOError as e:
                print(f"错误: 无法读取文件 '{file_name}'. 原因: {e}")
            except Exception as e:
                print(f"加载文件 '{file_name}' 时发生未知错误: {e}")
                
        self.dataframe = pd.DataFrame(datas)

    def is_data_loaded(self):
        """Checks if data was successfully loaded."""
        return not self.dataframe.empty
    
    def primary_metrics_visualization(self):   
        if not self.is_data_loaded():
            print("Error: No data loaded, cannot plot.")
            return

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"输出目录: {self.output_dir}")

        # 找出所有以 'val/test_score/' 开头且以 '_noformatf1' 结尾的列（即 F1 指标）
        f1_columns = [col for col in self.dataframe.columns if col.startswith('val/test_score/') and col.endswith('_noformatf1')]

        if len(f1_columns) == 0:
            print("Warning: No F1 score metrics found in the data.")
            return

        # 为每个 benchmark 单独绘图
        for col in f1_columns:
            # 提取 benchmark 名称，例如 val/test_score/tq_noformatf1 -> tq
            benchmark_name = col.split('/')[-1].replace('_noformatf1', '')

            plt.figure(figsize=(10, 6))
            plt.plot(self.dataframe[col], color='blue', linewidth=2, label=f"{benchmark_name}_ours")
            plt.xlabel("Training Steps")
            plt.ylabel("F1 Score")
            plt.title(f"F1 Score Evolution - {benchmark_name.upper()}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # 保存为单独的图片
            save_path = os.path.join(self.output_dir, f"{benchmark_name}_f1_over_steps.png")
            plt.savefig(save_path)
            plt.close()

            print(f"Saved F1 plot for {benchmark_name} to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Metrics Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认输出目录 (base_dir/figures)
  python metrics_visualization.py --base_dir ./eval_log

  # 指定自定义输出目录
  python metrics_visualization.py --base_dir ./eval_log --output_dir ./my_figures

  # 使用环境变量
  export EVAL_LOG_DIR=./eval_log
  python metrics_visualization.py --base_dir $EVAL_LOG_DIR
        """
    )
    parser.add_argument(
        '--base_dir', 
        type=str, 
        default=os.environ.get('EVAL_LOG_DIR', './eval_log'),
        help="包含 metric_step_*.json 文件的目录 (默认: $EVAL_LOG_DIR 或 ./eval_log)"
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=None,
        help="保存图片的输出目录 (默认: base_dir/figures)"
    )

    args = parser.parse_args()

    print(f"读取目录: {args.base_dir}")
    
    visualizer = MetricsVisualizer(
        base_dir=args.base_dir,
        output_dir=args.output_dir
    )

    # Check if data loaded successfully
    if not visualizer.is_data_loaded():
        print("Error: No data loaded.")
        sys.exit(1)
    
    visualizer.primary_metrics_visualization()
    print("可视化完成!")


if __name__ == "__main__":
    main()
