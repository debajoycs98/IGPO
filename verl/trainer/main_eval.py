# Copyright 2024 IGPO Team
# Licensed under the Apache License, Version 2.0
"""
IGPO Multi-turn Dialogue Evaluation Script

This is a dedicated evaluation script for IGPO that only loads necessary 
components for evaluation, without training-related overhead.

Usage:
    python -m verl.trainer.main_eval \
        data.val_files=./data/test.parquet \
        model.path=/path/to/model \
        eval.output_dir=./eval_results \
        eval.max_turns=10
"""

import os
import json
from collections import defaultdict

import hydra
import numpy as np
import ray
from omegaconf import OmegaConf

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.dataset import RLHFDataset
from verl.utils.dataset.rl_dataset import collate_fn
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def compute_f1_score(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    import re
    import string
    
    def preprocess(text):
        text = text.lower()
        for punct in string.punctuation:
            text = text.replace(punct, ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # Extract answer from <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', prediction.lower(), re.DOTALL)
    if not answer_match:
        return 0.0
    pred_text = preprocess(answer_match.group(1))
    
    ground_truths = ground_truth.lower().split("<|answer_split|>")
    max_f1 = 0.0
    
    for gt in ground_truths:
        gt = preprocess(gt)
        pred_tokens = set(pred_text.split())
        gt_tokens = set(gt.split())
        
        if not gt_tokens or not pred_tokens:
            continue
            
        common = pred_tokens & gt_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            max_f1 = max(max_f1, f1)
    
    return max_f1


def compute_em_score(prediction: str, ground_truth: str) -> float:
    """Compute Exact Match score."""
    import re
    import string
    
    def preprocess(text):
        text = text.lower()
        for punct in string.punctuation:
            text = text.replace(punct, ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    answer_match = re.search(r'<answer>(.*?)</answer>', prediction.lower(), re.DOTALL)
    if not answer_match:
        return 0.0
    pred_text = preprocess(answer_match.group(1))
    
    ground_truths = ground_truth.lower().split("<|answer_split|>")
    for gt in ground_truths:
        if preprocess(gt) == pred_text:
            return 1.0
    return 0.0


@hydra.main(config_path="config", config_name="eval", version_base=None)
def main(config):
    """Main evaluation entry point."""
    print("=" * 60)
    print("IGPO Evaluation Script")
    print("=" * 60)
    
    # Print config
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(config))
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
        )
    
    # Run evaluation
    ray.get(run_evaluation.remote(config))


@ray.remote(num_cpus=1)
def run_evaluation(config):
    """Run the evaluation process."""
    from torch.utils.data import DataLoader
    
    # Load tokenizer
    local_model_path = copy_to_local(config.model.path)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create rollout worker group
    print("\n[1/4] Initializing model...")
    ray_cls = RayClassWithInitArgs(
        cls=ray.remote(ActorRolloutRefWorker),
        config=config,
        role="rollout"
    )
    resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes
    )
    rollout_wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls)
    rollout_wg.init_model()
    
    # Load validation dataset
    print("\n[2/4] Loading validation dataset...")
    val_dataset = RLHFDataset(
        data_files=config.data.val_files,
        tokenizer=tokenizer,
        config=config.data,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    
    # Import generation manager
    from scrl.llm_agent.generation import LLMGenerationManager, GenerationConfig
    
    gen_config = GenerationConfig(
        max_turns=config.eval.max_turns,
        num_gpus=config.trainer.n_gpus_per_node,
        data_writing_path=config.eval.get("data_writing_path", None),
        model_name=config.model.path,
        n=1,
        search_engine=config.eval.search_engine,
        nnodes=config.trainer.nnodes,
        oss_access_key_id=config.data.get("oss_access_key_id", ""),
        oss_access_key_secret=config.data.get("oss_access_key_secret", ""),
        oss_endpoint=config.data.get("oss_endpoint", ""),
        codeact_env_disabled=config.eval.get("codeact_env_disabled", True),
        info_gain_type=config.eval.get("info_gain_type", "prob_diff"),
    )
    
    # Create MessageClient if needed
    client = None
    if config.eval.get("data_writing_path"):
        from tools_server.util import MessageClient
        client = MessageClient(
            config.eval.data_writing_path,
            isconsumer=True,
            oss_access_key_id=config.data.get("oss_access_key_id", ""),
            oss_access_key_secret=config.data.get("oss_access_key_secret", ""),
            oss_endpoint=config.data.get("oss_endpoint", ""),
        )
    
    generation_manager = LLMGenerationManager(
        tokenizer=tokenizer,
        actor_rollout_wg=rollout_wg,
        config=gen_config,
        is_validation=True,
        client=client,
    )
    
    # Run evaluation
    print("\n[3/4] Running evaluation...")
    results = []
    metrics_by_source = defaultdict(lambda: {"f1": [], "em": []})
    
    total_batches = len(val_dataloader)
    for batch_idx, batch_dict in enumerate(val_dataloader):
        print(f"\nProcessing batch {batch_idx + 1}/{total_batches}...")
        
        test_batch = DataProto.from_single_dict(batch_dict)
        
        # Get ground truths
        ground_truths = [
            {"ground_truth": x.non_tensor_batch["reward_model"]["ground_truth"]} 
            for x in test_batch
        ]
        data_sources = test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(test_batch))
        
        # Prepare generation batch
        test_gen_batch = test_batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
        test_gen_batch.meta_info = {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": False,
            "validate": True,
        }
        
        # Normalize batch size
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if len(test_gen_batch) >= n_gpus:
            norm_len = len(test_gen_batch) // n_gpus * n_gpus
        else:
            norm_len = len(test_gen_batch)
        
        test_gen_batch = test_gen_batch[:norm_len]
        ground_truths = ground_truths[:norm_len]
        data_sources = data_sources[:norm_len]
        
        # Generate
        _, final_output, info_gain_rewards = generation_manager.run_llm_loop(
            gen_batch=test_gen_batch,
            global_steps=-1,
            ground_truths=ground_truths,
        )
        
        # Compute metrics
        for i in range(len(final_output)):
            response = tokenizer.decode(final_output.batch["responses"][i], skip_special_tokens=False)
            gt = ground_truths[i]["ground_truth"]
            data_source = data_sources[i]
            
            f1 = compute_f1_score(response, gt)
            em = compute_em_score(response, gt)
            
            metrics_by_source[data_source]["f1"].append(f1)
            metrics_by_source[data_source]["em"].append(em)
            
            results.append({
                "data_source": data_source,
                "ground_truth": gt,
                "response": response,
                "f1_score": f1,
                "em_score": em,
                "info_gain_rewards": info_gain_rewards[i] if i < len(info_gain_rewards) else [],
            })
    
    # Aggregate metrics
    print("\n[4/4] Computing final metrics...")
    final_metrics = {}
    for source, scores in metrics_by_source.items():
        final_metrics[f"{source}/f1"] = np.mean(scores["f1"])
        final_metrics[f"{source}/em"] = np.mean(scores["em"])
    
    # Overall metrics
    all_f1 = [r["f1_score"] for r in results]
    all_em = [r["em_score"] for r in results]
    final_metrics["overall/f1"] = np.mean(all_f1)
    final_metrics["overall/em"] = np.mean(all_em)
    final_metrics["total_samples"] = len(results)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for key, value in sorted(final_metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Save results
    output_dir = config.eval.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(output_dir, "eval_results.jsonl")
    with open(results_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nDetailed results saved to: {results_file}")
    
    # Save metrics
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)
    
    return final_metrics


if __name__ == "__main__":
    main()
