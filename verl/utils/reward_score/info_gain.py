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

def compute_score(solution_str, ground_truth, data_source, val_type='f1', info_gain_reward=[], tokenizer=None, is_validation=False):
    if tokenizer is None:
        raise ValueError("tokenizer cannot be None")
        
    alpha = 1.0

    if is_validation:
        f1_score = compute_f1(solution_str, ground_truth, data_source, val_type='f1')
        em_score = compute_f1(solution_str, ground_truth, data_source, val_type='em')
        noformatf1_score = compute_f1(solution_str, ground_truth, data_source, val_type='noformatf1')
    else:
        f1_score = compute_f1(solution_str, ground_truth, data_source, val_type)
    
    all_tokens = tokenizer.tokenize(solution_str)
    scores = [0.0] * len(all_tokens)  # Use float array

    start_index = 0
    end_index = 0
    tokens_size = len(all_tokens)

    chats = solution_str.split("\n<|im_start|>assistant\n")
    chats_size = len(chats)

   

    if info_gain_reward == [] or chats_size == 1:
        scores[-1] = alpha * f1_score
        
        if is_validation:
            return {
                "f1": f1_score,
                "em": em_score,
                "noformatf1": noformatf1_score,
                "scores": scores,
            }
        else:
            return scores
    if len(info_gain_reward) != chats_size - 1:
        if chats_size > 10:
            info_gain_reward.append(0.0)
        else:
            with open("/ossfs/workspace/linyang/FactAgent/DeepResearcher/info_gain.py_debug.json", 'a') as f:
                json.dump({'solution_str': solution_str, 'chats_size': chats_size, 'info_gain_len': len(info_gain_reward), 'info_gain_reward': info_gain_reward, 'chats': chats}, f)
            print("info_gain.py: turn error")
            scores[-1] = alpha * f1_score
            if is_validation:
                return {
                    "f1": f1_score,
                    "em": em_score,
                    "noformatf1": noformatf1_score,
                    "scores": scores,
                }
            else:
                return scores


    for i, chat in enumerate(chats):
        if i == 0:
            chat_len = len(tokenizer.tokenize(chat))
            end_index = start_index + chat_len
            num_tokens_in_slice = end_index - start_index
            if num_tokens_in_slice > 0:
                scores[start_index:end_index] = [0] * num_tokens_in_slice
                scores[end_index - 1] = info_gain_reward[i]
        else:
            if i < chats_size - 1:
                chat_len = len(tokenizer.tokenize(chat))
                start_index = len(tokenizer.tokenize(solution_str.split(chat)[0]))
                end_index = start_index + chat_len
                num_tokens_in_slice = end_index - start_index
                if num_tokens_in_slice > 0:
                    scores[start_index:end_index] = [0] * num_tokens_in_slice
                    scores[end_index - 1] = info_gain_reward[i]
            else:
                scores[-1] = alpha * f1_score
    # with open("/ossfs/workspace/linyang/FactAgent/DeepResearcher/info_gain.py.jsonl", 'a') as f:
    #     f.write(json.dumps({"f1": f1_score, "info_gain_reward": info_gain_reward, "scores": scores}) + '\n')
    if is_validation:
        return {
            "f1": f1_score,
            "em": em_score,
            "noformatf1": noformatf1_score,
            "scores": scores,
        }
    # import math		
    # for s in scores:
    #     if math.isnan(s):
    #         raise ValueError("Find nan in info_gain.py")
    return scores
