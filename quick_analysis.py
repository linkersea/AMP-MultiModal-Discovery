import pandas as pd
import numpy as np
from collections import Counter

# 读取数据
df = pd.read_csv('results_three_methods_20250626_211751/final_predicted_candidates_20250626_211751.csv')

print('=== 统计分析结果 ===')
print(f'总序列数: {len(df)}')
print(f'独特序列数: {df["sequence"].nunique()}')

# 各方法统计
methods = df['method'].value_counts()
print('\n各方法数量:')
for method, count in methods.items():
    print(f'  {method}: {count}')

# 活性分布
high_activity = len(df[df['predicted_activity'] > 0.9])
perfect_activity = len(df[df['predicted_activity'] == 1.0])
print('\n活性分布:')
print(f'  高活性(>0.9): {high_activity} ({high_activity/len(df)*100:.1f}%)')
print(f'  完美预测(1.0): {perfect_activity} ({perfect_activity/len(df)*100:.1f}%)')

# 长度统计
print('\n长度统计:')
print(f'  平均长度: {df["length"].mean():.1f}')
print(f'  长度范围: {df["length"].min():.0f}-{df["length"].max():.0f}')

# 各方法详细分析
print('\n各方法详细分析:')
for method in methods.index:
    method_df = df[df['method'] == method]
    avg_activity = method_df['predicted_activity'].mean()
    high_count = len(method_df[method_df['predicted_activity'] > 0.9])
    perfect_count = len(method_df[method_df['predicted_activity'] == 1.0])
    print(f'  {method}:')
    print(f'    平均活性: {avg_activity:.4f}')
    print(f'    高活性比例: {high_count/len(method_df)*100:.1f}%')
    print(f'    完美预测: {perfect_count}个')

# 氨基酸分析
print('\n氨基酸组成分析:')
all_sequences = ''.join(df['sequence'].tolist())
aa_counts = Counter(all_sequences)
total_aa = len(all_sequences)

important_aa = ['K', 'R', 'F', 'W', 'Y']
for aa in important_aa:
    count = aa_counts.get(aa, 0)
    percentage = count / total_aa * 100
    print(f'  {aa}: {percentage:.1f}%')
