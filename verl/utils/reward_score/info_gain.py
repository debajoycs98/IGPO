from openai import OpenAI
import re
import difflib
import string
import json
import os

# 导入严格验证模块
try:
    from verl.utils.debug.igpo_pipeline_checker import (
        is_strict_check_enabled,
        record_info_gain_assignment,
    )
    _HAS_STRICT_CHECK = True
except ImportError:
    _HAS_STRICT_CHECK = False
    def is_strict_check_enabled(): return False

# 导入完整验证模块
try:
    from verl.utils.debug.igpo_full_checker import (
        is_full_check_enabled,
        get_full_checker,
    )
    _HAS_FULL_CHECK = True
except ImportError:
    _HAS_FULL_CHECK = False
    def is_full_check_enabled(): return False

def check_tags_balance(solution_str: str) -> bool:
    """检查标签是否正确配对
    
    Args:
        solution_str: 需要检查的字符串
    
    Returns:
        bool: 标签是否都正确配对
    """
    # 需要检查的标签对
    tags_to_check = ['code', 'tool_call', 'think', 'answer']
    
    for tag in tags_to_check:
        # 计算开始和结束标签的数量
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        
        start_count = solution_str.count(start_tag)
        end_count = solution_str.count(end_tag)
        
        # 如果开始和结束标签数量不相等，返回False
        if start_count != end_count:
            return False
            
        # 检查标签的嵌套顺序（确保结束标签不会在开始标签之前出现）
        last_pos = -1
        while True:
            start_pos = solution_str.find(start_tag, last_pos + 1)
            if start_pos == -1:
                break
                
            end_pos = solution_str.find(end_tag, start_pos)
            if end_pos == -1:
                return False
                
            last_pos = end_pos
            
    return True

def preprocess_text(text: str) -> str:
    """预处理文本，用于数据集的评分
    
    处理步骤:
    1. 转换为小写
    2. 移除标点符号 (.,!?;:'"()[]{}...)
    3. 去除多余空格
    """
    # 将标点符号替换为空格
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    
    # 替换多个空格为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    # 去除首尾空格
    text = text.strip()
    return text

def deal_multi_labels(ground_truth):
    for item in ground_truth:
        if item['label'].lower() == 'false':
            return 'false'
    return 'true'



def compute_f1(solution_str, ground_truth, data_source, val_type='f1') -> float:
    if data_source in ['Factbench', 'politifact', 'liar2']:
        ground_truth = json.loads(ground_truth)
        ground_truth = deal_multi_labels(ground_truth)
    solution_str = solution_str.lower()
    ground_truth = ground_truth.lower()
    ground_truths = ground_truth.split("<|answer_split|>")
    # 首先检查标签是否配对正确(格式是否正确)
    if not check_tags_balance(solution_str):
        
        if val_type == 'noformatf1':
            return 0
        else:
            return -2.0

    # 使用正则提取第一个<answer>标签中的内容
    try:
        answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            # 对答案进行预处理
            answer_content = preprocess_text(answer_content)
        else:
            if val_type == 'noformatf1':
                return 0
            else:
                return -2.0
    except Exception as e:
        print(f"Error extracting answer content: {e}")
        if val_type == 'noformatf1':
            return 0
        else:
            return -2.0
    
    max_score = 0.0
    
    for gt in ground_truths:
        # 对ground truth进行预处理
        gt = preprocess_text(gt)

        if val_type == 'em':
            if gt == answer_content:
                return 1.0
        else:
            # 将答案和参考答案分词
            pred_tokens = set(answer_content.split())
            gt_tokens = set(gt.split())
            
            if not gt_tokens:  # 避免除零错误
                continue
            if not pred_tokens:
                continue
            
            # 计算共同的词数
            common_tokens = pred_tokens & gt_tokens
            
            # 计算精确率和召回率
            precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
            recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
            
            # 计算F1分数
            if precision + recall > 0:  # 避免除零错误
                f1 = 2 * (precision * recall) / (precision + recall)
                max_score = max(max_score, f1)
            
    return max_score

def _char_pos_to_token_idx(char_pos, offset_mapping):
    """
    根据字符位置找到对应的 token 索引
    
    Args:
        char_pos: 字符位置
        offset_mapping: tokenizer 返回的 offset_mapping，格式为 [(start, end), ...]
    
    Returns:
        对应的 token 索引
    """
    for i, (start, end) in enumerate(offset_mapping):
        if start <= char_pos < end:
            return i
        if char_pos < start:
            # char_pos 在当前 token 之前的空隙中，返回前一个 token
            return max(0, i - 1)
    # 如果超出范围，返回最后一个 token
    return len(offset_mapping) - 1


def compute_score(solution_str, ground_truth, data_source, val_type='f1', info_gain_reward=[], tokenizer=None, is_validation=False):
    """
    计算 token 级别的奖励分数
    
    使用 tokenizer 的 offset_mapping 功能实现精确的 token-字符位置映射，
    避免 subword tokenization 导致的索引计算错误。
    
    Args:
        solution_str: 模型生成的完整响应字符串
        ground_truth: 标准答案
        data_source: 数据来源
        val_type: 验证类型 ('f1', 'em', 'noformatf1')
        info_gain_reward: 每个 turn 的信息增益奖励列表
        tokenizer: HuggingFace tokenizer
        is_validation: 是否为验证模式
    
    Returns:
        scores 列表或包含多个指标的字典
    """
    import os
    debug_pipeline = os.environ.get("DEBUG_IGPO_PIPELINE", "0") == "1"
    
    if tokenizer is None:
        raise ValueError("tokenizer cannot be None")
        
    alpha = 1.0

    # 计算 F1/EM 分数
    if is_validation:
        f1_score = compute_f1(solution_str, ground_truth, data_source, val_type='f1')
        em_score = compute_f1(solution_str, ground_truth, data_source, val_type='em')
        noformatf1_score = compute_f1(solution_str, ground_truth, data_source, val_type='noformatf1')
    else:
        f1_score = compute_f1(solution_str, ground_truth, data_source, val_type)
    
    # 使用 offset_mapping 获取精确的 token-字符位置映射
    encoding = tokenizer(solution_str, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = encoding['input_ids']
    offset_mapping = encoding['offset_mapping']  # [(char_start, char_end), ...]
    
    tokens_size = len(token_ids)
    scores = [0.0] * tokens_size

    # 如果没有 token，直接返回空 scores
    if tokens_size == 0:
        if is_validation:
            return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1_score, "scores": scores}
        return scores

    # Turn 分隔符
    separator = "\n<|im_start|>assistant\n"
    
    # 找到所有 turn 的起始字符位置
    turn_start_positions = []  # 每个 turn 内容的起始字符位置
    turn_end_positions = []    # 每个 turn 内容的结束字符位置
    
    # 查找所有分隔符位置
    sep_positions = []
    search_pos = 0
    while True:
        sep_pos = solution_str.find(separator, search_pos)
        if sep_pos == -1:
            break
        sep_positions.append(sep_pos)
        search_pos = sep_pos + 1
    
    if len(sep_positions) == 0:
        # 没有分隔符，整个字符串视为一个 turn
        turn_start_positions = [0]
        turn_end_positions = [len(solution_str)]
    else:
        # 第一个分隔符之前的内容（如果有）
        if sep_positions[0] > 0:
            turn_start_positions.append(0)
            turn_end_positions.append(sep_positions[0])
        
        # 每个分隔符之后的 turn
        for i, sep_pos in enumerate(sep_positions):
            turn_start = sep_pos + len(separator)
            turn_start_positions.append(turn_start)
            
            # 确定这个 turn 的结束位置
            if i + 1 < len(sep_positions):
                turn_end = sep_positions[i + 1]
            else:
                turn_end = len(solution_str)
            turn_end_positions.append(turn_end)
    
    chats_size = len(turn_start_positions)
    
    # ========== 完整验证：记录 Turn 边界 ==========
    full_check = _HAS_FULL_CHECK and is_full_check_enabled()
    if full_check:
        checker = get_full_checker()
        sample_idx = getattr(compute_score, '_sample_counter', 0)
        
        # 记录字符级别的 turn 边界
        char_boundaries = [(turn_start_positions[i], turn_end_positions[i]) for i in range(chats_size)]
        
        # 计算预期的 token 位置（每个 turn 的最后一个 token）
        expected_positions = []
        for i in range(chats_size):
            turn_end_char = turn_end_positions[i]
            if turn_end_char > 0:
                expected_pos = _char_pos_to_token_idx(turn_end_char - 1, offset_mapping)
            else:
                expected_pos = 0
            expected_pos = min(expected_pos, tokens_size - 1)
            expected_positions.append(expected_pos)
        
        # Token 边界（基于 offset_mapping）
        token_boundaries = []
        for i in range(chats_size):
            start_char = turn_start_positions[i]
            end_char = turn_end_positions[i]
            start_token = _char_pos_to_token_idx(start_char, offset_mapping)
            end_token = _char_pos_to_token_idx(end_char - 1, offset_mapping) if end_char > 0 else 0
            token_boundaries.append((start_token, end_token))
        
        checker.record_turn_boundaries(sample_idx, char_boundaries, token_boundaries)

    # 如果没有 info_gain_reward 或只有一个 turn，只在最后一个 token 放 f1_score
    if info_gain_reward == [] or chats_size == 1:
        scores[-1] = alpha * f1_score
        
        # 即使没有 info_gain，也需要记录验证数据
        if full_check:
            compute_score._sample_counter = sample_idx + 1
            record_info_gain_assignment(
                sample_idx=sample_idx,
                info_gain_rewards=[],  # 没有 info_gain
                info_gain_positions=[],
                f1_score=alpha * f1_score,
                f1_position=tokens_size - 1,
            )
        
        if is_validation:
            return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1_score, "scores": scores}
        return scores
    
    # 检查 info_gain_reward 长度是否匹配
    if len(info_gain_reward) != chats_size - 1:
        print(f"info_gain.py: turn mismatch - chats_size={chats_size}, info_gain_len={len(info_gain_reward)}")
        # 长度不匹配时，回退到只使用 f1_score
        scores[-1] = alpha * f1_score
        
        # 即使 turn mismatch，也需要记录验证数据
        if full_check:
            compute_score._sample_counter = sample_idx + 1
            record_info_gain_assignment(
                sample_idx=sample_idx,
                info_gain_rewards=[],  # 没有 info_gain
                info_gain_positions=[],
                f1_score=alpha * f1_score,
                f1_position=tokens_size - 1,
            )
        
        if is_validation:
            return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1_score, "scores": scores}
        return scores

    # ========== DEBUG: 验证点 1 & 2 - info_gain_reward 计算和分配 ==========
    reward_assignments = []  # 用于记录奖励分配信息
    
    # 为每个 turn 的最后一个 token 分配奖励
    for i in range(chats_size):
        turn_end_char = turn_end_positions[i]
        
        # 找到该 turn 最后一个字符对应的 token 索引
        # 注意：turn_end_char 是开区间，所以用 turn_end_char - 1
        if turn_end_char > 0:
            last_token_idx = _char_pos_to_token_idx(turn_end_char - 1, offset_mapping)
        else:
            last_token_idx = 0
        
        # 确保索引在有效范围内
        last_token_idx = min(last_token_idx, tokens_size - 1)
        
        # 分配奖励
        if i < chats_size - 1:
            ig_value = info_gain_reward[i]
            if ig_value == 0.0:
                ig_value = 1e-10  # 避免被 !=0 检查跳过
            scores[last_token_idx] = ig_value
            reward_assignments.append({
                'turn': i,
                'type': 'info_gain',
                'value': ig_value,
                'token_idx': last_token_idx,
                'turn_range': (turn_start_positions[i], turn_end_positions[i]),
            })
        else:
            scores[last_token_idx] = alpha * f1_score
            reward_assignments.append({
                'turn': i,
                'type': 'f1',
                'value': alpha * f1_score,
                'token_idx': last_token_idx,
                'turn_range': (turn_start_positions[i], turn_end_positions[i]),
            })
    
    # ========== 完整验证：记录实际的 Token 位置 ==========
    if full_check:
        actual_positions = [ra['token_idx'] for ra in reward_assignments]
        # expected_positions 已在前面计算
        checker.record_reward_positions(sample_idx, actual_positions, expected_positions)
        
        # 记录 rewards 用于传输验证
        ig_rewards = [ra['value'] for ra in reward_assignments if ra['type'] == 'info_gain']
        f1_value = [ra['value'] for ra in reward_assignments if ra['type'] == 'f1']
        all_rewards = ig_rewards + f1_value
        checker.rewards_at_info_gain[sample_idx] = all_rewards
    
    # ========== DEBUG 输出 ==========
    if debug_pipeline:
        print(f"\n[IGPO Pipeline Check 1 & 2] === info_gain.py: Reward Assignment ===")
        print(f"  Solution length: {len(solution_str)} chars, {tokens_size} tokens")
        print(f"  Total turns: {chats_size}")
        print(f"  Input info_gain_reward: {info_gain_reward}")
        print(f"  F1 score: {f1_score:.4f}")
        print(f"\n  Reward Assignments:")
        for ra in reward_assignments:
            turn_content_preview = solution_str[ra['turn_range'][0]:min(ra['turn_range'][0]+50, ra['turn_range'][1])]
            turn_end_preview = solution_str[max(0, ra['turn_range'][1]-30):ra['turn_range'][1]]
            print(f"    Turn {ra['turn']} ({ra['type']}): value={ra['value']:.6f}, token_idx={ra['token_idx']}")
            print(f"      Turn range: chars [{ra['turn_range'][0]}, {ra['turn_range'][1]})")
            print(f"      Turn start: '{turn_content_preview}...'")
            print(f"      Turn end:   '...{turn_end_preview}'")
            
            # 验证 token 位置是否在 turn 范围内
            if ra['token_idx'] < len(offset_mapping):
                token_char_start, token_char_end = offset_mapping[ra['token_idx']]
                token_text = solution_str[token_char_start:token_char_end]
                in_range = ra['turn_range'][0] <= token_char_start < ra['turn_range'][1]
                print(f"      Token at idx {ra['token_idx']}: chars [{token_char_start}, {token_char_end}), text='{token_text}'")
                print(f"      Token in turn range: {in_range} {'✓' if in_range else '⚠️ WARNING: Token outside turn range!'}")
        
        # 统计非零 scores 数量
        nonzero_count = sum(1 for s in scores if s != 0)
        print(f"\n  Verification:")
        print(f"    Non-zero scores: {nonzero_count}")
        print(f"    Expected non-zero: {chats_size} (= {chats_size - 1} info_gain + 1 f1)")
        print(f"    Match: {nonzero_count == chats_size} {'✓' if nonzero_count == chats_size else '⚠️ MISMATCH!'}")
    
    # ========== 严格验证：记录分配信息 ==========
    strict_check = _HAS_STRICT_CHECK and is_strict_check_enabled()
    if strict_check:
        # 从 reward_assignments 中提取信息
        ig_rewards = []
        ig_positions = []
        f1_score_final = 0.0
        f1_position = -1
        
        for ra in reward_assignments:
            if ra['type'] == 'info_gain':
                ig_rewards.append(ra['value'])
                ig_positions.append(ra['token_idx'])
            else:  # f1
                f1_score_final = ra['value']
                f1_position = ra['token_idx']
        
        # 使用 data_source 或序号作为 sample_idx
        # 注意：这里无法直接获取 sample_idx，使用静态计数器
        sample_idx = getattr(compute_score, '_sample_counter', 0)
        compute_score._sample_counter = sample_idx + 1
        
        record_info_gain_assignment(
            sample_idx=sample_idx,
            info_gain_rewards=ig_rewards,
            info_gain_positions=ig_positions,
            f1_score=f1_score_final,
            f1_position=f1_position,
        )
    
    if is_validation:
        return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1_score, "scores": scores}
    return scores
