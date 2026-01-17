import pandas as pd
import sys

# 获取文件路径，默认使用本地文件
if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    file_path = './test_v4.parquet'  # 默认文件路径

# 使用 pyarrow 引擎读取 parquet 文件
df = pd.read_parquet(file_path, engine='pyarrow')

print(f"文件: {file_path}")
print(f"总行数: {len(df)}")
print(f"列名: {list(df.columns)}")
print()

# 统计每个 data_source 的数量
if 'data_source' in df.columns:
    print("=" * 50)
    print("各 data_source 的数据条数:")
    print("=" * 50)
    
    data_source_counts = df['data_source'].value_counts()
    
    for source, count in data_source_counts.items():
        print(f"  {source}: {count}")
    
    print("=" * 50)
    print(f"共有 {len(data_source_counts)} 种不同的 data_source")
else:
    print("⚠️ 该文件中没有 'data_source' 列")
    print("可用的列名:", list(df.columns))
