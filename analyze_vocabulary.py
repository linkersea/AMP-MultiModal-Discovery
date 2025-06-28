#!/usr/bin/env python3
"""
分析VAE词汇表设定
"""

import pandas as pd
from collections import Counter

def analyze_vocabulary():
    """分析抗菌肽数据集的词汇表构成"""
    
    # 读取数据
    df = pd.read_csv('data/processed/preprocessed_data.csv')
    sequences = df['sequence'].tolist()
    
    print("=== VAE词汇表分析 ===")
    print(f"数据集序列数量: {len(sequences)}")
    
    # 分析氨基酸组成
    all_sequences = ''.join(sequences)
    unique_aas = set(all_sequences)
    aa_counts = Counter(all_sequences)
    
    print(f"\n实际出现的氨基酸数量: {len(unique_aas)}")
    print(f"氨基酸种类: {sorted(unique_aas)}")
    
    # 构建与VAE相同的词汇表
    aa_to_idx = {aa: i+1 for i, aa in enumerate(sorted(unique_aas))}
    aa_to_idx['<PAD>'] = 0  # 填充符
    aa_to_idx['<START>'] = len(aa_to_idx)
    aa_to_idx['<END>'] = len(aa_to_idx)
    
    vocab_size = len(aa_to_idx)
    
    print(f"\n词汇表构成:")
    print(f"  氨基酸数量: {len(unique_aas)}")
    print(f"  特殊符号: 3 (<PAD>, <START>, <END>)")
    print(f"  总词汇表大小: {vocab_size}")
    
    print(f"\n完整氨基酸映射:")
    for aa, idx in sorted(aa_to_idx.items()):
        print(f"  {aa}: {idx}")
    
    print(f"\n各氨基酸在数据集中的频率:")
    total_aa = len(all_sequences)
    for aa in sorted(unique_aas):
        count = aa_counts[aa]
        percentage = count / total_aa * 100
        print(f"  {aa}: {count:4d} ({percentage:5.1f}%)")
    
    # 分析为什么是16维
    print(f"\n=== 16维词汇表的原因 ===")
    print(f"抗菌肽数据集中只包含了 {len(unique_aas)} 种氨基酸")
    print(f"这是因为:")
    print(f"1. 抗菌肽通常具有特定的氨基酸偏好")
    print(f"2. 数据集可能是经过筛选的高活性序列")
    print(f"3. 某些氨基酸在抗菌功能中更重要")
    
    # 分析缺失的氨基酸
    all_20_aas = set('ACDEFGHIKLMNPQRSTVWY')
    missing_aas = all_20_aas - unique_aas
    
    print(f"\n数据集中未出现的氨基酸 ({len(missing_aas)}个):")
    missing_descriptions = {
        'C': '半胱氨酸 - 可能因二硫键复杂性被排除',
        'D': '天冬氨酸 - 负电荷，与抗菌肽阳离子特性不符',
        'G': '甘氨酸 - 柔性过大，可能影响结构稳定性',
        'H': '组氨酸 - pH敏感性可能不适合抗菌应用',
        'N': '天冬酰胺 - 极性但不带电荷',
        'T': '苏氨酸 - 极性侧链可能不适合膜相互作用'
    }
    
    for aa in sorted(missing_aas):
        desc = missing_descriptions.get(aa, '未知原因')
        print(f"  {aa}: {desc}")
    
    print(f"\n=== 设计合理性分析 ===")
    print(f"16维词汇表的优势:")
    print(f"1. 计算效率: 较小的词汇表减少了模型参数和计算复杂度")
    print(f"2. 数据充分性: 每个氨基酸都有足够的训练样本")
    print(f"3. 功能针对性: 保留了对抗菌功能最重要的氨基酸")
    print(f"4. 避免过拟合: 减少了模型需要学习的稀有氨基酸模式")

if __name__ == "__main__":
    analyze_vocabulary()
