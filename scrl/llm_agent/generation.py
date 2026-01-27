# =============================================================================
# Based on the Search-R1 example from the Search-R1 project.
#
# Original Authors: Jin Bowen, Zeng Hansi, Yue Zhenrui, Wang Dong, Zamani Hamed, Han Jiawei
#
# License: Apache 2.0
# Project URL: https://github.com/PeterGriffinJin/Search-R1
# =============================================================================

import torch
import copy
import re
import os
import json
from typing import List, Dict, Tuple, Optional

# Debug flag for shape printing (set IGPO_DEBUG_SHAPES=true to enable)
DEBUG_SHAPES = os.environ.get('IGPO_DEBUG_SHAPES', '').lower() in ('true', '1', 'yes')
if DEBUG_SHAPES:
    print("[IGPO] Debug mode enabled: tensor shape printing is ON (IGPO_DEBUG_SHAPES=true)")

# Import vectorized GT logprob module
from scrl.llm_agent.vectorized_gt_logprob import (
    is_vectorized_enabled,
    VectorizedGTConfig,
    VectorizedGTLogProbComputer,
)
import math
from dataclasses import dataclass
from tensordict import TensorDict
from scrl.llm_agent.tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
import numpy as np
import traceback
import torch.nn.functional as F
from tools_server.util import MessageClient
from tools_server.initialize_prompts import SYSTEM_PROMPT


@dataclass
class GenerationConfig:
    max_turns: int
    num_gpus: int
    data_writing_path: str = None
    model_name: str = None
    n: int = 1
    project_name: str = None
    experiment_name: str = None
    search_engine: str = "rag"
    nnodes: int = 1
    oss_access_key_id: str = ''
    oss_access_key_secret: str = ''
    oss_endpoint: str = ''
    system_prompt: Optional[str] = SYSTEM_PROMPT
    codeact_env_disabled: bool = True
    # info_gain_type: "prob_diff" (概率差) 或 "log_prob_diff" (log概率差)
    info_gain_type: str = "prob_diff"
    

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
        client = None,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.system_prompt = config.system_prompt
        self.codeact_env_disabled = config.codeact_env_disabled

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id
        ))

        self.client = client

    def _update_right_side(self, original_right_side: Dict, 
                           cur_responses: torch.Tensor,
                           next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side of rollings."""
        if next_obs_ids is not None:
            responses = self.tensor_fn.concatenate_with_padding(
                [original_right_side['responses'], cur_responses, next_obs_ids],
                pad_to_left=False
            )
        else:
            responses = self.tensor_fn.concatenate_with_padding(
                [original_right_side['responses'], cur_responses],
                pad_to_left=False
            )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        
        return {'responses': responses[:, :effective_len]}

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, next_obs_ids: torch.Tensor) -> DataProto:
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        return DataProto.from_dict({
                'input_ids': new_input_ids[:, -effective_len:],
                'position_ids': new_position_ids[:, -effective_len:],
                'attention_mask': new_attention_mask[:, -effective_len:]
            })

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']
        return next_obs_ids
        
    def postprocess_predictions(self, rollings_active: DataProto, gen_output: DataProto) -> Tuple[List[int], List[bool]]:
        """Postprocess predictions to remove padding and convert to list of strings."""
        """return: list of query contents including history"""

        pass
        return [{"prompt":""} for _ in range(rollings_active.batch['input_ids'].shape[0])]


    def execute_predictions(
        self, tool_call_list, total_number
    ) :
        query_contents = [{"idx": tool_call[0], "question": tool_call[1], "think": tool_call[2],
                           "tool_call": tool_call[3], "total_number":total_number} for tool_call in tool_call_list]

        query_contents = self.client.submit_tasks(query_contents)
        return query_contents

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus * self.config.nnodes
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        if remainder == 0:
            output = self.actor_rollout_wg.generate_sequences(active_batch)
            return output
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
        padded_active_batch = DataProto.from_dict(padded_batch)

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output



    def parse_question(self, input_ids: torch.Tensor) -> str:
        """Parse question to get the query content."""
        query_contents = self.tokenizer.batch_decode(input_ids)
        query_contents = [re.sub(r'^(<\|endoftext\|>)+', '', content) for content in query_contents]
        query_contents = [content.split("<|im_start|>user\n")[1].split("<|im_end|>")[0] for content in query_contents]
        return query_contents

    def parse_response(self, input_ids: torch.Tensor, think: bool = False) -> List[Tuple[bool, str, str]]:
        """Parse response to get the thinking process and answer or tool call.
            return: [(is_stop, thinking, answer/tool_call), ...]
        """
        response_contents = self.tokenizer.batch_decode(input_ids)
        results = []
        for i, content in enumerate(response_contents):
            if think:
                content = "<think>" + content
            if "<think>" in content and "<answer>" in content:
                if "</think>" not in content or "</answer>" not in content:
                    results.append((True, "", ""))
                else:
                    think = content.split("<think>")[1].split("</think>")[0]
                    answer = content.split("<answer>")[1].split("</answer>")[0]
                    results.append((True, think, answer))
            elif "<think>" in content and "<tool_call>" in content and self.codeact_env_disabled:
                if "</tool_call>" not in content or "</think>" not in content:
                    results.append((True, "", ""))
                else:
                    think = content.split("<think>")[1].split("</think>")[0]
                    tool_call = content.split("<tool_call>")[1].split("</tool_call>")[0]
                    try:
                        tool_call = json.loads(tool_call)
                        assert "name" in tool_call, "no vliad function name in tool_call"
                        assert "arguments" in tool_call, "no valid arguments in tool_call"
                        assert tool_call["name"] not in [""], "invalid tool name"
                        results.append((False, think, tool_call))
                    except Exception as e:
                        if i < 10:
                            print(f"model tool call format error: {e}")
                            print(content.replace('<|endoftext|>',''))
                        results.append((True, "", ""))
            elif "<think>" in content and "<code>" in content and not self.codeact_env_disabled:  # code act格式
                if "</code>" not in content or "</think>" not in content:
                    results.append((True, "", ""))
                else:
                    think = content.split("<think>")[1].split("</think>")[0]
                    code = content.split("<code>")[1].split("</code>")[0]
                    try:
                        tool_call = {"name":"code_act", "arguments":{"code":code}}
                        results.append((False, think, tool_call))
                    except Exception as e:
                        if i < 10:
                            print(f"model tool call format error: {e}")
                            print(content.replace('<|endoftext|>',''))
                        results.append((True, "", ""))
            else:
                results.append((True, "", ""))
        return results

    def pseudo_generate_sequences(self, prompts, response):
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        batch_size = idx.size(0)
        eos_token_id = self.tokenizer.eos_token_id
        non_tensor_batch = prompts.non_tensor_batch
        response = pad_2d_list_to_length(response, self.tokenizer.pad_token_id, max_length=2000).to(idx.device)

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def run_llm_loop(self, gen_batch: DataProto, global_steps: int, ground_truths: list) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""


        node_rank = int(os.environ["PET_NODE_RANK"])
        print(f"node {node_rank} gains {len(gen_batch.batch['input_ids'])} * {self.config.n} datas!",flush=True)
        query_contents = self.parse_question(gen_batch.batch['input_ids'])
        messages_list = []
        agent_grpo_idx = []
        for gt in ground_truths:
            if "<|answer_split|>" in gt['ground_truth']:
                gt['ground_truth'] = gt['ground_truth'].split("<|answer_split|>")[0]
            _gt = gt['ground_truth'].strip()
            if _gt.startswith('['):
                parsed_gt = json.loads(_gt)
                if isinstance(parsed_gt, list):
                    label = 'true'
                    for item in parsed_gt:
                        if item['label'].lower() == 'false':
                            label = 'false'
                            break
                    gt['ground_truth'] = label

        # 预处理ground_truths 对齐batch_size * n
        ground_truths_rolling = []
        for gt in ground_truths:
            for _ in range(self.config.n):
                ground_truths_rolling.append(gt)

        # 使用 offset_mapping 精确计算 ground truth 的 token 范围
        # 避免 subword tokenization 边界效应导致的索引偏差
        PREFIX = "\nNow there's enough information to answer\n</think>\n<answer>\n"
        SUFFIX = "\n</answer><|im_end|>"
        
        pseudo_resps_with_gt = []
        gt_idx = []
        
        for ground_truth in ground_truths_rolling:
            gt_text = ground_truth['ground_truth']
            full_text = f"{PREFIX}{gt_text}{SUFFIX}"
            
            # 使用 offset_mapping 获取精确的字符-token 映射
            encoding = self.tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
            token_ids = encoding['input_ids'].tolist()[0]
            offset_mapping = encoding['offset_mapping'].tolist()[0]  # [(char_start, char_end), ...]
            
            pseudo_resps_with_gt.append(token_ids)
            
            if len(token_ids) == 0:
                print(f"❗❗❗ EMPTY token_ids for ground_truth: '{gt_text}'")
                gt_idx.append([0, 0])
                continue
            
            # 计算 ground truth 在原始字符串中的位置
            gt_char_start = len(PREFIX)
            gt_char_end = len(PREFIX) + len(gt_text)
            
            # 通过 offset_mapping 找到精确的 token 索引
            gt_token_start = None
            gt_token_end = None
            
            for token_idx, (char_start, char_end) in enumerate(offset_mapping):
                # 找到第一个覆盖 gt_char_start 的 token
                if gt_token_start is None and char_end > gt_char_start:
                    gt_token_start = token_idx
                # 找到最后一个覆盖 gt 内容的 token（char_start < gt_char_end）
                if char_start < gt_char_end and char_end > 0:
                    gt_token_end = token_idx + 1
            
            # 边界检查
            if gt_token_start is None:
                gt_token_start = len(token_ids)
            if gt_token_end is None:
                gt_token_end = len(token_ids)
            
            gt_idx.append([gt_token_start, gt_token_end])
        

        for idx, query_content in enumerate(query_contents):
            for _ in range(self.config.n):
                if self.system_prompt:
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": query_content}
                    ]
                else:
                    messages = [
                        {"role": "user", "content": query_content}
                    ]
                messages_list.append(messages)
                agent_grpo_idx.append(idx)
        activate_list = [i for i in range(len(messages_list))]
        message_string_list = ["" for _ in range(len(messages_list))]

        # 确保保存目录存在
        output_dir = f"./outputs/{self.config.project_name}/{self.config.experiment_name}/rollout"
        if not os.path.exists(output_dir):
            print(f"Directory not exist, create at {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # 创建information gain reward：list[list]
        gt_log_probs_per_turn = [[] for _ in range(len(messages_list))]
        gt_entropys_per_turn = [[] for _ in range(len(messages_list))]
        info_gain_rewards = [[] for _ in range(len(messages_list))]
        gt_values = {}  # 存储上一轮的值（概率或log概率，取决于 info_gain_type）

        # 向量化开关检测
        use_vectorized_gt_logprob = is_vectorized_enabled()
        # 记录每个样本每个 turn 结束时的 token 位置（用于向量化计算）
        turn_end_positions_per_sample = [[] for _ in range(len(messages_list))] if use_vectorized_gt_logprob else None

        for step in range(self.config.max_turns):
            print(f"node {node_rank} step {step} start!")
            activate_messages_list = [messages_list[i] for i in activate_list]

            if activate_list == []:
                break
            try:
                rollings_active = self.tokenizer.apply_chat_template(activate_messages_list, add_generation_prompt=True, tokenize=False)
            except Exception as e:
                print(f"Error in tokenizer.apply_chat_template: {e}")
                json.dump(activate_messages_list, open('./debug.json', 'w'))
                # 回退策略：逐个处理每条消息
                rollings_active = []
                for msg in activate_messages_list:
                    try:
                        result = self.tokenizer.apply_chat_template([msg], add_generation_prompt=True, tokenize=False)
                        rollings_active.extend(result)
                    except Exception as inner_e:
                        print(f"Failed to process message: {inner_e}")
                        print(f"Message content: {msg}")
                        raise  # 无法恢复，抛出异常
    
            think = True
            
            if think:
                rollings_active = [rolling + "<think>" for rolling in rollings_active]
            else:
                rollings_active = [rolling for rolling in rollings_active]
            
            rollings_active = self.tokenizer(rollings_active, return_tensors="pt", padding=True)
                
            pad_mask = rollings_active['input_ids'] != self.tokenizer.pad_token_id
            sorted_indices = pad_mask.to(torch.int64).argsort(dim=1, stable=True)
            rollings_active['input_ids'] = rollings_active['input_ids'].gather(1, sorted_indices)
            rollings_active['attention_mask'] = rollings_active['attention_mask'].gather(1, sorted_indices)
            
            attention_mask = rollings_active['attention_mask']
            rollings_active['position_ids'] = self.tensor_fn.create_position_ids(attention_mask)

            print(f"node {node_rank}, turn {step} rollings_active is {len(rollings_active['input_ids'])} datas")
            rollings_active = DataProto.from_dict({
                'input_ids': rollings_active['input_ids'],
                'attention_mask': rollings_active['attention_mask'],
                'position_ids': rollings_active['position_ids'],
            })
            
            if step == 0:
                info_gain_rollings_active = copy.deepcopy(rollings_active)
            else:
                if info_gain_rollings_active.batch['input_ids'].shape[1] < rollings_active.batch['input_ids'].shape[1]:
                    info_gain_rollings_active.batch['input_ids'] = F.pad(
                        info_gain_rollings_active.batch['input_ids'], 
                        pad=(0, rollings_active.batch['input_ids'].shape[1] - info_gain_rollings_active.batch['input_ids'].shape[1]),
                        mode='constant',
                        value=self.tokenizer.pad_token_id,
                       )
					
                    info_gain_rollings_active.batch['attention_mask'] = F.pad(
                        info_gain_rollings_active.batch['attention_mask'], 
                        pad=(0, rollings_active.batch['attention_mask'].shape[1] - info_gain_rollings_active.batch['attention_mask'].shape[1]),
                        mode='constant',
                        value=0,  # attention_mask 用 0 填充表示 padding 位置
                        )

                    info_gain_rollings_active.batch['position_ids'] = F.pad(
                        info_gain_rollings_active.batch['position_ids'], 
                        pad=(0, rollings_active.batch['position_ids'].shape[1] - info_gain_rollings_active.batch['position_ids'].shape[1]),
                        mode='constant',
                        value=0,  # position_ids 用 0 填充
                        )
                
                    for i in range(len(activate_list)):
                        info_gain_rollings_active.batch['input_ids'][activate_list[i], :] = rollings_active.batch['input_ids'][i]
                        info_gain_rollings_active.batch['attention_mask'][activate_list[i], :] = rollings_active.batch['attention_mask'][i]
                        info_gain_rollings_active.batch['position_ids'][activate_list[i], :] = rollings_active.batch['position_ids'][i]
                else:
                    for i in range(len(activate_list)):   
                        info_gain_rollings_active.batch['input_ids'][activate_list[i], :len(rollings_active.batch['input_ids'][i])] = rollings_active.batch['input_ids'][i]
                        info_gain_rollings_active.batch['attention_mask'][activate_list[i], :len(rollings_active.batch['attention_mask'][i])] = rollings_active.batch['attention_mask'][i]
                        info_gain_rollings_active.batch['position_ids'][activate_list[i], :len(rollings_active.batch['position_ids'][i])] = rollings_active.batch['position_ids'][i]    
            
            
            pseudo_gen_output = self.pseudo_generate_sequences(info_gain_rollings_active, pseudo_resps_with_gt)

            # Debug shape printing (enable with IGPO_DEBUG_SHAPES=true)
            if DEBUG_SHAPES:
                print("rollings_active input_ids shape:", rollings_active.batch['input_ids'].shape)
                print("rollings_active attention_mask shape:", rollings_active.batch['attention_mask'].shape)
                print("rollings_active position_ids shape:", rollings_active.batch['position_ids'].shape)
                print("pseudo_gen_output prompts shape:", pseudo_gen_output.batch['prompts'].shape)
                print("info_gain_rollings_active input_ids shape:", info_gain_rollings_active.batch['input_ids'].shape)
                print("info_gain_rollings_active attention_mask shape:", info_gain_rollings_active.batch['attention_mask'].shape)
                print("info_gain_rollings_active position_ids shape:", info_gain_rollings_active.batch['position_ids'].shape)
                print("pseudo_gen_output prompts shape:", pseudo_gen_output.batch['prompts'].shape)
                print("pseudo_gen_output attention_mask shape:", pseudo_gen_output.batch['attention_mask'].shape)
                print("pseudo_gen_output position_ids shape:", pseudo_gen_output.batch['position_ids'].shape)
                print("pseudo_gen_output responses shape:", pseudo_gen_output.batch['responses'].shape)
                print("pseudo_gen_output input_ids shape:", pseudo_gen_output.batch['input_ids'].shape)
            
            pseudo_gen_output_log_probs = self.actor_rollout_wg.compute_log_prob(pseudo_gen_output)

            


            # with open("/ossfs/workspace/linyang/FactAgent/DeepResearcher/pseudo_gen_log_prob.json", 'w') as f:
            #     json.dump({"type": str(type(pseudo_gen_output_log_prob)), "pseudo_gen_output_log_prob": str(pseudo_gen_output_log_prob)}, f)
            # x = pseudo_gen_output_log_prob.batch['old_log_probs'].tolist()
            # import math
            # with open("/ossfs/workspace/linyang/FactAgent/DeepResearcher/pseudo_gen_log_prob_and_prob.jsonl", 'a') as f:
            #     for i in range(len(x)):
            #         json.dump({"log_prob_of_gt": [x[i][j] for j in range(gt_idx[i][0], gt_idx[i][1])],  
            #                    "prob_of_gt": [math.exp(x[i][j]) for j in range(gt_idx[i][0], gt_idx[i][1])]}, f)
            #         f.write('\n')
            
            
            # ========== 记录 turn 结束位置（用于向量化计算） ==========
            if use_vectorized_gt_logprob and turn_end_positions_per_sample is not None:
                # 记录每个活跃样本当前 turn 的 context 结束位置
                for idx, i in enumerate(activate_list):
                    # 从 attention_mask 计算当前样本的有效 token 数
                    valid_len = rollings_active.batch['attention_mask'][idx].sum().item()
                    turn_end_positions_per_sample[i].append(int(valid_len))
            
            # ========== 根据 info_gain_type 计算 info_gain_reward ==========
            # "prob_diff": 使用概率差 exp(mean(log P_t)) - exp(mean(log P_{t-1}))
            # "log_prob_diff": 使用 log 概率差 mean(log P_t) - mean(log P_{t-1})
            
            info_gain_type = self.config.info_gain_type  # "prob_diff" 或 "log_prob_diff"
            
            if step == 0:
                for i in activate_list:
                    log_probs = pseudo_gen_output_log_probs.batch['old_log_probs'][i, gt_idx[i][0]:gt_idx[i][1]]
                    mean_log_prob = log_probs.mean().item()
                    
                    if info_gain_type == "log_prob_diff":
                        # 存储 log 概率的均值
                        gt_values[i] = mean_log_prob
                    else:  # "prob_diff" (默认)
                        # 存储概率的几何平均（即 exp(mean(log P))）
                        gt_values[i] = torch.exp(torch.tensor(mean_log_prob)).item()
                    
                    gt_log_probs_per_turn[i].append(log_probs.tolist())
                    gt_entropys_per_turn[i].append(pseudo_gen_output_log_probs.batch['entropys'][i, gt_idx[i][0]:gt_idx[i][1]].tolist())
            else:
                for i in activate_list:
                    log_probs = pseudo_gen_output_log_probs.batch['old_log_probs'][i, gt_idx[i][0]:gt_idx[i][1]]
                    mean_log_prob = log_probs.mean().item()
                    
                    if info_gain_type == "log_prob_diff":
                        # 使用 log 概率差
                        cur_value = mean_log_prob
                        info_gain = cur_value - gt_values[i]
                    else:  # "prob_diff" (默认)
                        # 使用概率差
                        cur_value = torch.exp(torch.tensor(mean_log_prob)).item()
                        info_gain = cur_value - gt_values[i]
                    
                    info_gain_rewards[i].append(info_gain)
                    gt_values[i] = cur_value
                    
                    gt_log_probs_per_turn[i].append(log_probs.tolist())
                    gt_entropys_per_turn[i].append(pseudo_gen_output_log_probs.batch['entropys'][i, gt_idx[i][0]:gt_idx[i][1]].tolist())       

            gen_output = self._generate_with_gpu_padding(rollings_active)
            
            meta_info = gen_output.meta_info
            print(f"node {node_rank}, turn {step} gen_output {len(gen_output.batch['responses'])} datas")

            results = self.parse_response(gen_output.batch['responses'], think=think)
            assert len(results) == len(activate_list) # 每一轮更新后，结果数量和当前活跃的query数量一致
            activate_list_copy = []
            tool_call_list = []
            for i in range(len(results)):
                if results[i][0]:
                    message_string_list[activate_list[i]] = self.tokenizer.decode(rollings_active.batch['input_ids'][i], skip_special_tokens=False).replace("<|endoftext|>", "") + self.tokenizer.decode(gen_output.batch['responses'][i], skip_special_tokens=False).replace("<|endoftext|>", "")
                else:
                    activate_list_copy.append(activate_list[i])
                    tool_call_list.append((activate_list[i], messages_list[activate_list[i]][1]["content"], results[i][1], results[i][2]))
                    
            tool_call_list = self.execute_predictions(tool_call_list,len(messages_list))
            print(f"node {node_rank}, turn {step} tool_call_list {len(tool_call_list)} datas")
            for i in range(len(tool_call_list)):
                if not self.codeact_env_disabled:  # code act激活
                    messages_list[tool_call_list[i]['idx']].append(
                        {
                            "role": "assistant", 
                            "content": "<think>" + tool_call_list[i]['think'] + "</think>"+"\n<code>" + str(tool_call_list[i]['tool_call']['arguments']['code']) + "</code>", 
                        }
                    )
                    try:
                        messages_list[tool_call_list[i]['idx']].append(
                            {
                                "role": "user", 
                                "content": "<code_response>" + tool_call_list[i]['content'] + "</code_response>",
                            }
                        )
                    except:
                        messages_list[tool_call_list[i]['idx']].append(
                            {
                                "role": "user", 
                                "content": "<code_response>" + '返回格式错误，code执行失败' + "</code_response>",
                            }
                        )
                else:
                    messages_list[tool_call_list[i]['idx']].append(
                        {
                            "role": "assistant", 
                            "content": "<think>" + tool_call_list[i]['think'] + "</think>", 
                            "tool_calls": [
                                            {
                                                "type": "function", 
                                                "function": tool_call_list[i]['tool_call']
                                            }
                                        ]
                        }
                    )
                    try:
                        messages_list[tool_call_list[i]['idx']].append(
                            {
                                "role": "tool", 
                                "name": tool_call_list[i]['tool_call']['name'],
                                "content": tool_call_list[i]['content']
                            }
                        )
                    except:
                        messages_list[tool_call_list[i]['idx']].append(
                            {
                                "role": "tool", 
                                "name": '',
                                "content": '返回格式错误，tool调用失败'
                            }
                        )
            print(f"第{step}轮结束， node {node_rank} 原本有{len(activate_list)}个query，现在有{len(activate_list_copy)}个query")
            activate_list = activate_list_copy
            # with open(f"/ossfs/workspace/linyang/FactAgent/DeepResearcher/messages_list_{step}.jsonl", 'a') as f:
            #     for message in messages_list:
            #         json.dump(message, f)
            #         f.write('\n')
        
        # 保存 gt_log_probs 到本地输出目录（如果需要调试，可取消注释）
        # gt_log_probs_path = os.path.join(output_dir, f"gt_log_probs_{global_steps}.json")
        # with open(gt_log_probs_path, 'w') as f:
        #     json.dump({"gt_log_probs_per_turn": gt_log_probs_per_turn, "gt_entropys_per_turn": gt_entropys_per_turn}, f)
        
        # ========== 向量化 GT LogProb 状态报告 ==========
        if use_vectorized_gt_logprob and turn_end_positions_per_sample is not None:
            # 统计 turn 位置信息
            total_turns = sum(len(pos) for pos in turn_end_positions_per_sample)
            samples_with_turns = sum(1 for pos in turn_end_positions_per_sample if len(pos) > 0)
            
            print(f"[IGPO] Vectorized GT LogProb ENABLED: "
                  f"{samples_with_turns}/{len(messages_list)} samples, "
                  f"{total_turns} total turns recorded")
            
            # 打印 info_gain 统计
            total_info_gains = sum(len(r) for r in info_gain_rewards)
            print(f"[IGPO] Info gain rewards computed: {total_info_gains} values")

        if activate_list != []:
            for i in activate_list:
                message_string_list[i] = self.tokenizer.apply_chat_template(messages_list[i], add_generation_prompt=True, tokenize=False)
        
        response_str_list = []
        initial_prompt_list = []
        for i, messages in enumerate(messages_list):
            initial_prompt = self.tokenizer.apply_chat_template(messages[0:2], add_generation_prompt=True, tokenize=False)
            initial_prompt_list.append(initial_prompt)
            response_str_list.append(message_string_list[i][len(initial_prompt):])
        
        prompts_tokenizered = self.tokenizer(initial_prompt_list, return_tensors="pt",padding=True)

        prompts_repeated = prompts_tokenizered['input_ids']
        pad_mask = prompts_repeated != self.tokenizer.pad_token_id
        sorted_indices = pad_mask.to(torch.int64).argsort(dim=1, stable=True)
        

        prompts_repeated = prompts_repeated.gather(1, sorted_indices)
        prompts_attention_mask = prompts_tokenizered['attention_mask'].gather(1, sorted_indices)

        responses = self.tokenizer(response_str_list, return_tensors="pt",padding=True)['input_ids']
        
        responses_attention_mask = self.tokenizer(response_str_list, return_tensors="pt",padding=True)['attention_mask']
        attention_mask = torch.cat((prompts_attention_mask, responses_attention_mask), dim=-1)
        position_ids = self.tensor_fn.create_position_ids(attention_mask)
        
        message_tensor = DataProto.from_dict({
            'prompts': prompts_repeated,
            'responses': responses,
            'input_ids': torch.cat((prompts_repeated, responses), dim=-1),
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        })
        message_tensor.meta_info.update(meta_info)
        message_tensor.non_tensor_batch['agent_grpo_idx'] = np.array(agent_grpo_idx, dtype=object)
        print("generation结束")


        # import math
        print(f"node {node_rank} message_string_list {len(message_string_list)}")
        # for info_gain_reward in info_gain_rewards:
        #     for r in info_gain_reward:
        #         if math.isnan(r):
        #             raise ValueError("Find nan in info_gain_rewards!")
        return message_string_list, message_tensor, info_gain_rewards
    
