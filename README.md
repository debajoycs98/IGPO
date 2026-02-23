<h1 align="center">✨ Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn Search Agents

</h1>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<div align="center"> 

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2510.14967)
[![Paper](https://img.shields.io/badge/🤗%20Paper-Hugging%20Face-yellow)](https://huggingface.co/papers/2510.14967)
[![Ant Group](https://img.shields.io/badge/Supported%20by-Ant%20Group-1677FF.svg?logo=ant-design)](https://www.antgroup.com/)

</div>

## 📣 Latest News
- **[Feb 01, 2026]**: 🔄 Codebase updated with new features. [See details](resources/CHANGELOG_20260201.md)
- **[Jan 26, 2026]**: 🎉 Our **[IGPO](https://arxiv.org/abs/2510.14967)** paper has been accepted at **ICLR 2026**!
- **[Oct 17, 2025]**: 📄 Our IGPO paper is now available on **[arXiv](https://arxiv.org/abs/2510.14967)** and **[Hugging Face](https://huggingface.co/papers/2510.14967)** daily paper.

## 💡 Method Overview
We introduce IGPO, a RL algorithm for fine-grained credit assignment in search agent training. By modeling agentic search turns as an incremental information acquisition process, IGPO defines rewards as the marginal gain in the policy's probability of generating the correct answer. 

<p align="center">
    <img src="./images/Framework.png" width="100%">
</p>

## 📊 Overall Performance
<p align="center">
    <img src="./images/Exp.png" width="100%">
</p>

## 🚀 Quick Start

### 1. Installation

> **Tested on**: 8x NVIDIA A100-SXM4-80GB, CUDA 12.4, Driver 580.x, Python 3.10

#### 1.1 Create conda environment

```bash
git clone https://github.com/debajoycs98/IGPO.git
cd IGPO

conda create -n igpo python=3.10 -y
conda activate igpo
```

#### 1.2 Install PyTorch (CUDA 12.4)

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
```

#### 1.3 Install vLLM and Flash Attention

```bash
pip install vllm==0.6.3.post1
pip install flash-attn==2.8.3 --no-build-isolation
```

#### 1.4 Install remaining dependencies

```bash
pip install -r requirements.txt
```

#### 1.5 Install verl (this repo) in editable mode

```bash
pip install -e . --no-deps
```

> **Note**: `--no-deps` prevents pip from overriding the pinned versions of torch/vllm installed above.

#### 1.6 Install supplementary packages

```bash
pip install pyvers antlr4-python3-runtime==4.9.3 sentry-sdk platformdirs gitpython
```

#### 1.7 Pin transformers to 4.x

The codebase is not yet compatible with `transformers` 5.x. Pin to 4.x:

```bash
pip install 'transformers>=4.43.0,<5.0.0'
```

#### 1.8 Apply vLLM 0.6.3.post1 compatibility patches

The verl codebase requires 6 small patches to work with `vllm==0.6.3.post1`. Apply them as follows:

<details>
<summary><b>Patch 1: <code>verl/third_party/vllm/__init__.py</code></b> — recognize version string <code>0.6.3.post1</code></summary>

```diff
- elif package_version == "0.6.3" or package_version == "0.6.3+rocm624" or package_version == "0.6.3+rocm634":
+ elif package_version in ("0.6.3", "0.6.3.post1", "0.6.3+rocm624", "0.6.3+rocm634"):
```

</details>

<details>
<summary><b>Patch 2: <code>verl/workers/rollout/vllm_rollout/__init__.py</code></b> — use semantic version comparison</summary>

```diff
- if package_version <= "0.6.3":
+ from packaging import version as vs
+
+ if package_version and vs.parse(package_version) <= vs.parse("0.6.3.post1"):
```

</details>

<details>
<summary><b>Patch 3: <code>verl/workers/sharding_manager/fsdp_vllm.py</code></b> — fix model_runner access for SPMDGPUExecutor</summary>

```diff
- self.model_runner = inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner if inference_engine else None
+ if inference_engine is None:
+     self.model_runner = None
+ else:
+     executor = inference_engine.llm_engine.model_executor
+     if hasattr(executor, "driver_worker"):
+         self.model_runner = executor.driver_worker.worker.model_runner
+     else:
+         self.model_runner = executor.worker.model_runner
```

</details>

<details>
<summary><b>Patch 4: <code>verl/utils/vllm_utils.py</code></b> — make DeepseekV3ForCausalLM import optional</summary>

```diff
- from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM
+ from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
+ try:
+     from vllm.model_executor.models.deepseek_v2 import DeepseekV3ForCausalLM
+ except ImportError:
+     DeepseekV3ForCausalLM = None
  from vllm.model_executor.models.qwen2_moe import Qwen2MoeForCausalLM
- model_types = [Qwen2MoeForCausalLM, DeepseekV2ForCausalLM, DeepseekV3ForCausalLM]
+ model_types = [Qwen2MoeForCausalLM, DeepseekV2ForCausalLM]
+ if DeepseekV3ForCausalLM is not None:
+     model_types.append(DeepseekV3ForCausalLM)
```

</details>

<details>
<summary><b>Patch 5: <code>verl/workers/fsdp_workers.py</code></b> — make vLLMAsyncRollout import conditional</summary>

```diff
- from verl.workers.rollout.vllm_rollout import vllm_mode, vLLMAsyncRollout, vLLMRollout
+ from verl.workers.rollout.vllm_rollout import vllm_mode, vLLMRollout
+ try:
+     from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout
+ except ImportError:
+     vLLMAsyncRollout = None
```

</details>

<details>
<summary><b>Patch 6: <code>verl/third_party/vllm/vllm_v_0_6_3/llm_engine_sp.py</code></b> — handle removed attribute</summary>

```diff
- scheduler_config.use_v2_block_manager,
+ getattr(scheduler_config, 'use_v2_block_manager', True),
```

</details>

### 2. Configure Web Search API & Prompt Template

Edit `tools_server/config.yaml`:

```yaml
# Google Search (via Serper API)
search_engine: "google"
serper_api_key: "your_serper_api_key_here"

# Or Bing Search (via Azure)
# search_engine: "bing"
# azure_bing_search_subscription_key: "your_bing_key_here"

# System prompt template is also defined in this file
# system_prompt: |-
#   ...
```

> **💡 Tip**: If you don't have access to a search API yet, you can use **mock mode** for testing:
> ```bash
> export IGPO_MOCK_SEARCH=true
> ```
> Or set `mock_mode: true` in `tools_server/config.yaml`. This will return simulated search results without actual API calls.

### 3. Prepare Data

Place your training data in the `data/` directory:
- `data/train.parquet` - Training data
- `data/dev.parquet` - Validation data
- `data/test.parquet` - Evaluation data

Data format: See the provided data for reference.

### 4. Training

> **Supported Models**: We currently support Qwen series models (e.g., Qwen2.5-7B-Instruct).

Edit `train.sh` to configure training parameters, then run:

```bash
bash train.sh
```

<details>
<summary><b>Key Parameters in <code>train.sh</code></b></summary> 

| Parameter | Description |
|-----------|-------------|
| `MODEL_PATH` | Path to your model or Hugging Face model name (e.g., `Qwen/Qwen2.5-7B-Instruct`) |
| `OUTPUT` | Directory for saving checkpoints |
| `EVAL_LOG_PATH` | Directory for saving validation results |
| `data.train_files` | Path to training data (parquet format) |
| `data.val_files` | Path to validation data (parquet format) |
| `algorithm.gamma` | Discount factor for reward computation |
| `+algorithm.info_gain_type` | Info gain reward calculation: `log_prob_diff` (log probability difference) or `prob_diff` (probability difference) |
| `+algorithm.info_gain_norm_mode` | Reward normalization: `separate` (normalize info gain and outcome rewards independently) or `joint` (normalize all rewards together) |
| `+algorithm.use_vectorized_gt_logprob` | Enable vectorized ground truth log probability computation for faster info gain reward construction |
| `+algorithm.use_curriculum` | Enable curriculum learning to gradually decay info gain reward weight during training |
| `+algorithm.curriculum_f1_init/final` | Initial and final weight for outcome reward in curriculum learning |
| `+algorithm.curriculum_ig_init/final` | Initial and final weight for info gain reward in curriculum learning |
| `trainer.save_freq` | Save checkpoint every N training steps |
| `trainer.test_freq` | Run validation every N training steps |
| `agent_grpo.n` | Number of rollouts per sample (GRPO group size) |
| `max_turns` | Maximum number of search turns allowed per episode |

</details>

### 5. Evaluation

Edit `evaluate.sh` to configure evaluation parameters, then run:

```bash
bash evaluate.sh
```

<details>
<summary><b>Key Parameters in <code>evaluate.sh</code></b></summary>

| Parameter | Description |
|-----------|-------------|
| `MODEL_PATH` | Path to your trained checkpoint |
| `TEST_FILES` | Path to test data (parquet format) |
| `OUTPUT_DIR` | Directory for saving evaluation results |
| `EVAL_LOG_PATH` | Directory for saving evaluation logs |
| `MAX_TURNS` | Maximum number of search turns allowed per episode |

</details>

## 📄 Citation
If you find our code or work useful for your research, please cite our work.
```bibtex
@inproceedings{wang2026information,
  title={Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn Search Agents},
  author={Wang, Guoqing and Dai, Sunhao and Ye, Guangze and Gan, Zeyu and Yao, Wei and Deng, Yong and Wu, Xiaofeng and Ying, Zhenzhe},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

## 🙏 Acknowledgement 

IGPO is inspired by [Deepseek-R1](https://github.com/deepseek-ai/DeepSeek-R1), with its implementation built upon [veRL](https://github.com/volcengine/verl), [Search-r1](https://github.com/PeterGriffinJin/Search-R1), and [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher). We are grateful to the teams behind these projects for their significant contributions to open-source research and development.

## 📞 Contact

For any questions or feedback, please reach out to us at guoqingwang905@gmail.com.

## 📜 License

This project is released under the [MIT License](LICENSE).

## 🌟 Star History

<a href="https://star-history.com/#GuoqingWang1/IGPO&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=GuoqingWang1/IGPO&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=GuoqingWang1/IGPO&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=GuoqingWang1/IGPO&type=Date" />
 </picture>
</a>
