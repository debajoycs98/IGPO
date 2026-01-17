from openai import OpenAI
import re
import difflib
import string
import json

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

    # 如果没有 info_gain_reward 或只有一个 turn，只在最后一个 token 放 f1_score
    if info_gain_reward == [] or chats_size == 1:
        scores[-1] = alpha * f1_score
        if is_validation:
            return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1_score, "scores": scores}
        return scores
    
    # 检查 info_gain_reward 长度是否匹配
    if len(info_gain_reward) != chats_size - 1:
        print(f"info_gain.py: turn mismatch - chats_size={chats_size}, info_gain_len={len(info_gain_reward)}")
        # 长度不匹配时，回退到只使用 f1_score
        scores[-1] = alpha * f1_score
        if is_validation:
            return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1_score, "scores": scores}
        return scores

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
        else:
            scores[last_token_idx] = alpha * f1_score
    
    if is_validation:
        return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1_score, "scores": scores}
    return scores
