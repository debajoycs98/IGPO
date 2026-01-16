import pandas as pd

# 读取 parquet 文件
df = pd.read_parquet('/ossfs/workspace/linyang/FactAgent/DeepResearcher/data/test_v4.parquet')

# 转换为 JSON
df.to_json('test_v4.json', orient='records', lines=True)  # 每行一个 JSON 对象
# 或者