import pandas as pd

# 读取数据
df = pd.read_csv('results_three_methods_20250626_211751/final_predicted_candidates_20250626_211751.csv')

print('=== 各方法Top 5序列 ===')
methods = ['sequence_variation', 'rational_design', 'vae_generation']

for method in methods:
    method_df = df[df['method'] == method].sort_values('predicted_activity', ascending=False)
    print(f'\n{method}:')
    for i, (idx, row) in enumerate(method_df.head(5).iterrows()):
        print(f'  {i+1}. {row["sequence"]} (评分: {row["predicted_activity"]:.4f})')
