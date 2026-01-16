import pandas as pd

# 使用 pyarrow 引擎，更高效
df = pd.read_parquet('/ossfs/workspace/linyang/FactAgent/DeepResearcher/data/test_v3.parquet', engine='pyarrow')
print(f"该 Parquet 文件共有 {len(df)} 行数据")
print(len(df) // 16)
print(len(df) // 16 * 16)