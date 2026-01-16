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
    def __init__(self, base_dir="/ossfs/workspace/linyang/FactAgent/DeepResearcher/eval_log"):
        # baseline_data_dir = "/ossfs/workspace/linyang/FactAgent/DeepResearcher/eval_log_baseline_rag"
        self.base_dir = base_dir

        # baseline_file_path_list = os.listdir(baseline_data_dir)
        file_path_list = os.listdir(base_dir)

        metric_files = [f for f in file_path_list if f.startswith('metric_step_') and f.endswith('.json')]
        # baseline_metric_files = [f for f in baseline_file_path_list if f.startswith('metric_step_') and f.endswith('.json')]
        try:
            # baseline_metric_files.sort(key=lambda f: int(f.replace('metric_step_', '').replace('.json', '')))
            metric_files.sort(key=lambda f: int(f.replace('metric_step_', '').replace('.json', '')))
        except ValueError:
            print("Error: Could not parse step numbers from filenames. Please ensure they follow the 'metric_step_x.json' format.")
            self.dataframe = pd.DataFrame() # Create an empty dataframe
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

        # baseline_datas = []
        # for file_name in baseline_metric_files:
        #     file_path = os.path.join(baseline_data_dir, file_name)
        #     try:
        #         with open(file_path, 'r', encoding='utf-8') as f: 
        #             if 'metric' in file_name:
        #                 baseline_data = json.load(f)
        #                 baseline_datas.append(baseline_data)
        #                 print(f"成功加载: {file_name}")
        #     except json.JSONDecodeError:
        #         print(f"错误: 文件 '{file_name}' 不是有效的 JSON 格式。")
        #     except IOError as e:
        #         print(f"错误: 无法读取文件 '{file_name}'. 原因: {e}")
        #     except Exception as e:
        #             print(f"加载文件 '{file_name}' 时发生未知错误: {e}")
        # self.dataframe = pd.DataFrame(datas)
        # self.baseline_dataframe = pd.DataFrame(baseline_datas)

    def is_data_loaded(self):
        """Checks if data was successfully loaded."""
        return not self.dataframe.empty
    
    def primary_metrics_visualization(self):   
        if not self.is_data_loaded():
            print("Error: No data loaded, cannot plot.")
            return

		# 确保输出目录存在
        output_dir = '/ossfs/workspace/linyang/FactAgent/DeepResearcher/eval_log_3B_ours/figures'
        os.makedirs(output_dir, exist_ok=True)

		# 找出所有以 'val/test_score/' 开头且以 '_noformatf1' 结尾的列（即 F1 指标）
        f1_columns = [col for col in self.dataframe.columns if col.startswith('val/test_score/') and col.endswith('_noformatf1')]

        if len(f1_columns) == 0:
            print("Warning: No F1 score metrics found in the data.")
            return

		# 为每个 benchmark 单独绘图
        for col in f1_columns:
			# 提取 benchmark 名称，例如 val/test_score/tq_noformatf1 -> tq
            if col not in ["val/test_score/browse_comp_noformatf1", "val/test_score/browse_comp_zh_noformatf1", "val/test_score/xbench_deepsearch_noformatf1"]:
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
                save_path = os.path.join(output_dir, f"{benchmark_name}_f1_over_steps.png")
                plt.savefig(save_path)
                plt.close()

                print(f"Saved F1 plot for {benchmark_name} to {save_path}")
            else:
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
                save_path = os.path.join(output_dir, f"{benchmark_name}_f1_over_steps.png")
                plt.savefig(save_path)
                plt.close()

                print(f"Saved F1 plot for {benchmark_name} to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Metrics Visualization")
    parser.add_argument('--base_dir', type=str, required=True, help="BASE_DIR")

    args = parser.parse_args()

    visualizer = MetricsVisualizer(args.base_dir)
    

    # Check if data loaded successfully
    if not visualizer.is_data_loaded():
        print("Error: No data loaded.")
        return
    
    visualizer.primary_metrics_visualization()

if __name__ == "__main__":
    main()    
