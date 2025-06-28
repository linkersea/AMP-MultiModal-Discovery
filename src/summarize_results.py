import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_ROOT = 'results/classification/'
SUMMARY_CSV = os.path.join(RESULTS_ROOT, 'all_experiment_summary.csv')
REPORT_MD = os.path.join(RESULTS_ROOT, 'all_experiment_report.md')

summary_rows = []

for subdir, dirs, files in os.walk(RESULTS_ROOT):
    if 'model_comparison.csv' in files:
        exp_name = os.path.relpath(subdir, RESULTS_ROOT)
        csv_path = os.path.join(subdir, 'model_comparison.csv')
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            summary_rows.append({
                'experiment': exp_name,
                'model': row['model'],
                'acc': row['acc'],
                'auc': row['auc'],
                'f1': row['f1'],
                'best_k': row['best_k'] if 'best_k' in row else None
            })

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values(['auc', 'f1', 'acc'], ascending=False)
summary_df.to_csv(SUMMARY_CSV, index=False)

# 可视化：各实验-模型性能热力图，显示best_k
pivot_auc = summary_df.pivot(index='experiment', columns='model', values='auc')
pivot_k = summary_df.pivot(index='experiment', columns='model', values='best_k')
plt.figure(figsize=(12, max(6, 0.5*len(pivot_auc))))
sns.heatmap(pivot_auc, annot=pivot_k, fmt='', cmap='YlGnBu', cbar_kws={'label': 'AUC'})
plt.title('AUC for Each Experiment/Model (best_k in cell)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_ROOT, 'auc_heatmap_bestk.png'))
plt.close()

pivot_f1 = summary_df.pivot(index='experiment', columns='model', values='f1')
plt.figure(figsize=(12, max(6, 0.5*len(pivot_f1))))
sns.heatmap(pivot_f1, annot=pivot_k, fmt='', cmap='YlOrRd', cbar_kws={'label': 'F1'})
plt.title('F1 Score for Each Experiment/Model (best_k in cell)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_ROOT, 'f1_heatmap_bestk.png'))
plt.close()

# 条形图：Top-N模型/实验，显示best_k
TOP_N = 10
plt.figure(figsize=(10, 6))
sns.barplot(data=summary_df.head(TOP_N), x='auc', y='experiment', hue='model')
for i, row in summary_df.head(TOP_N).iterrows():
    plt.text(row['auc'], i, f"k={int(row['best_k']) if pd.notnull(row['best_k']) else '-'}", va='center', ha='left', fontsize=9, color='black')
plt.title(f'Top {TOP_N} AUC Results (best_k shown)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_ROOT, 'top_auc_bar_bestk.png'))
plt.close()

plt.figure(figsize=(10, 6))
sns.barplot(data=summary_df.head(TOP_N), x='f1', y='experiment', hue='model')
for i, row in summary_df.head(TOP_N).iterrows():
    plt.text(row['f1'], i, f"k={int(row['best_k']) if pd.notnull(row['best_k']) else '-'}", va='center', ha='left', fontsize=9, color='black')
plt.title(f'Top {TOP_N} F1 Results (best_k shown)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_ROOT, 'top_f1_bar_bestk.png'))
plt.close()

# 自动生成markdown报告，突出显示best_k
with open(REPORT_MD, 'w', encoding='utf-8') as f:
    f.write('# 多肽抗菌活性分类批量实验报告\n\n')
    f.write('## 总体说明\n')
    f.write(f'- 共计 {len(summary_df)} 组模型/特征组合实验\n')
    f.write(f'- 结果已汇总于 `{SUMMARY_CSV}`\n')
    f.write(f'- 主要性能可视化见下图（每格内为AUC/F1，括号内为最优k）\n\n')
    f.write('![](auc_heatmap_bestk.png)\n')
    f.write('![](f1_heatmap_bestk.png)\n')
    f.write('![](top_auc_bar_bestk.png)\n')
    f.write('![](top_f1_bar_bestk.png)\n')
    f.write('\n## Top 10 实验结果（按AUC排序，含最优k）\n')
    f.write(summary_df.head(10).to_markdown(index=False))
    f.write('\n\n## Top 10 实验结果（按F1排序，含最优k）\n')
    f.write(summary_df.sort_values(['f1','auc','acc'],ascending=False).head(10).to_markdown(index=False))
    f.write('\n\n---\n')
    f.write('> 本报告由 summarize_results.py 自动生成\n')

print(f"汇总完成，已保存到: {SUMMARY_CSV}")
print(f"可视化与报告已生成于: {RESULTS_ROOT}")
print(summary_df.head(20))
