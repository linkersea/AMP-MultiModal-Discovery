#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于活性位点分析的抗菌多肽理性设计
分析训练数据中的关键位点和氨基酸组合
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import itertools
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """加载数据并进行基本分析"""
    df = pd.read_csv('data/raw/120dataset.csv')
    
    print("数据基本信息:")
    print(f"总序列数: {len(df)}")
    print(f"活性范围: {df['activity'].min():.3f} - {df['activity'].max():.3f}")
    print(f"平均活性: {df['activity'].mean():.3f}")
    
    return df

def position_specific_analysis(df):
    """位置特异性分析"""
    print("\n" + "="*40)
    print("位置特异性分析")
    print("="*40)
    
    # 对齐序列到相同长度进行位置分析
    sequences = df['sequence'].tolist()
    activities = df['activity'].tolist()
    
    # 分析不同长度的序列
    length_groups = defaultdict(list)
    for seq, act in zip(sequences, activities):
        length_groups[len(seq)].append((seq, act))
    
    print(f"序列长度分布:")
    for length, seqs in sorted(length_groups.items()):
        avg_activity = np.mean([act for _, act in seqs])
        print(f"长度 {length}: {len(seqs)} 个序列, 平均活性: {avg_activity:.3f}")
    
    # 对最常见的长度进行位置分析
    most_common_length = max(length_groups.keys(), key=lambda x: len(length_groups[x]))
    print(f"\n对长度为 {most_common_length} 的序列进行位置分析 (共{len(length_groups[most_common_length])}个)")
    
    target_sequences = [seq for seq, _ in length_groups[most_common_length]]
    target_activities = [act for _, act in length_groups[most_common_length]]
    
    # 计算每个位置的氨基酸活性贡献
    position_contributions = {}
    
    for pos in range(most_common_length):
        aa_activities = defaultdict(list)
        
        for seq, act in zip(target_sequences, target_activities):
            aa = seq[pos]
            aa_activities[aa].append(act)
        
        # 计算每个氨基酸在该位置的平均活性
        pos_contributions = {}
        for aa, acts in aa_activities.items():
            if len(acts) >= 2:  # 至少出现2次
                pos_contributions[aa] = {
                    'mean_activity': np.mean(acts),
                    'count': len(acts),
                    'std': np.std(acts)
                }
        
        position_contributions[pos] = pos_contributions
    
    # 显示每个位置的最佳氨基酸
    print(f"\n各位置最佳氨基酸:")
    optimal_sequence = ""
    for pos in range(most_common_length):
        if position_contributions[pos]:
            best_aa = max(position_contributions[pos].keys(), 
                         key=lambda aa: position_contributions[pos][aa]['mean_activity'])
            best_activity = position_contributions[pos][best_aa]['mean_activity']
            best_count = position_contributions[pos][best_aa]['count']
            optimal_sequence += best_aa
            print(f"位置 {pos+1:2d}: {best_aa} (平均活性: {best_activity:.3f}, 出现次数: {best_count})")
        else:
            optimal_sequence += "X"
    
    print(f"\n理论最优序列: {optimal_sequence}")
    
    return position_contributions, most_common_length, optimal_sequence

def motif_analysis(df):
    """motif分析"""
    print("\n" + "="*40)
    print("序列motif分析")
    print("="*40)
    
    sequences = df['sequence'].tolist()
    activities = df['activity'].tolist()
    
    # 分析2-4氨基酸的motif
    motif_activities = defaultdict(list)
    
    for motif_len in [2, 3, 4]:
        print(f"\n{motif_len}氨基酸motif分析:")
        
        # 提取所有motif
        for seq, act in zip(sequences, activities):
            for i in range(len(seq) - motif_len + 1):
                motif = seq[i:i+motif_len]
                motif_activities[motif].append(act)
        
        # 筛选出现频率>=3的motif
        frequent_motifs = {motif: acts for motif, acts in motif_activities.items() 
                          if len(acts) >= 3 and len(motif) == motif_len}
        
        # 按平均活性排序
        sorted_motifs = sorted(frequent_motifs.items(), 
                             key=lambda x: np.mean(x[1]), reverse=True)
        
        print(f"高活性{motif_len}氨基酸motif (top 10):")
        for motif, acts in sorted_motifs[:10]:
            mean_act = np.mean(acts)
            count = len(acts)
            std_act = np.std(acts)
            print(f"  {motif}: 平均活性 {mean_act:.3f} ± {std_act:.3f} (n={count})")
    
    return motif_activities

def amino_acid_preference_analysis(df):
    """氨基酸偏好性分析"""
    print("\n" + "="*40)
    print("氨基酸偏好性分析")
    print("="*40)
    
    # 计算每个氨基酸的活性贡献
    aa_contributions = defaultdict(list)
    
    for seq, act in zip(df['sequence'], df['activity']):
        for aa in seq:
            aa_contributions[aa].append(act)
    
    # 计算统计信息
    aa_stats = {}
    for aa, activities in aa_contributions.items():
        aa_stats[aa] = {
            'mean_activity': np.mean(activities),
            'median_activity': np.median(activities),
            'count': len(activities),
            'std': np.std(activities),
            'frequency': len(activities) / sum(len(seq) for seq in df['sequence'])
        }
    
    # 按平均活性排序
    sorted_aa = sorted(aa_stats.items(), key=lambda x: x[1]['mean_activity'], reverse=True)
    
    print("氨基酸活性贡献排序:")
    print(f"{'氨基酸':<4} {'平均活性':<8} {'中位活性':<8} {'频率':<8} {'出现次数':<8}")
    print("-" * 45)
    
    for aa, stats in sorted_aa:
        print(f"{aa:<4} {stats['mean_activity']:<8.3f} {stats['median_activity']:<8.3f} "
              f"{stats['frequency']:<8.3f} {stats['count']:<8}")
    
    return aa_stats

def design_optimized_peptides(position_contributions, motif_activities, aa_stats, target_length=12):
    """基于分析结果设计优化的多肽序列"""
    print("\n" + "="*40)
    print("基于分析结果的理性设计")
    print("="*40)
    
    # 获取高活性氨基酸
    high_activity_aa = [aa for aa, stats in aa_stats.items() 
                       if stats['mean_activity'] > np.mean([s['mean_activity'] for s in aa_stats.values()])]
    
    print(f"高活性氨基酸: {', '.join(sorted(high_activity_aa))}")
    
    # 获取高活性motif
    high_activity_motifs = []
    for motif, activities in motif_activities.items():
        if len(activities) >= 3 and np.mean(activities) > 0.5:  # 活性阈值
            high_activity_motifs.append((motif, np.mean(activities)))
    
    high_activity_motifs.sort(key=lambda x: x[1], reverse=True)
    top_motifs = [motif for motif, _ in high_activity_motifs[:20]]
    
    print(f"高活性motif数量: {len(top_motifs)}")
    print(f"Top 5 motif: {top_motifs[:5]}")
    
    # 设计策略1: 基于高活性氨基酸组合
    design1_sequences = []
    
    print(f"\n设计策略1: 高活性氨基酸随机组合")
    for i in range(20):
        sequence = ""
        for _ in range(target_length):
            # 按活性加权随机选择氨基酸
            weights = [aa_stats[aa]['mean_activity'] for aa in high_activity_aa]
            chosen_aa = np.random.choice(high_activity_aa, p=np.array(weights)/sum(weights))
            sequence += chosen_aa
        design1_sequences.append(sequence)
        print(f"  设计{i+1:2d}: {sequence}")
    
    # 设计策略2: 基于motif组合
    design2_sequences = []
    
    print(f"\n设计策略2: 高活性motif组合")
    for i in range(20):
        sequence = ""
        remaining_length = target_length
        
        while remaining_length > 0:
            # 选择适合的motif
            suitable_motifs = [motif for motif in top_motifs if len(motif) <= remaining_length]
            
            if suitable_motifs and np.random.random() < 0.7:  # 70%概率使用motif
                chosen_motif = np.random.choice(suitable_motifs)
                sequence += chosen_motif
                remaining_length -= len(chosen_motif)
            else:
                # 使用单个高活性氨基酸填充
                weights = [aa_stats[aa]['mean_activity'] for aa in high_activity_aa]
                chosen_aa = np.random.choice(high_activity_aa, p=np.array(weights)/sum(weights))
                sequence += chosen_aa
                remaining_length -= 1
        
        design2_sequences.append(sequence)
        print(f"  设计{i+1:2d}: {sequence}")
    
    # 设计策略3: 基于位置特异性
    design3_sequences = []
    
    if position_contributions:
        print(f"\n设计策略3: 基于位置特异性优化")
        base_length = len(position_contributions)
        
        for i in range(20):
            sequence = ""
            
            # 如果目标长度与分析长度不同，需要调整
            if target_length == base_length:
                # 直接使用位置信息
                for pos in range(target_length):
                    if pos in position_contributions and position_contributions[pos]:
                        # 从该位置的前3个最佳氨基酸中选择
                        pos_aa = position_contributions[pos]
                        sorted_pos_aa = sorted(pos_aa.keys(), 
                                             key=lambda aa: pos_aa[aa]['mean_activity'], reverse=True)
                        chosen_aa = np.random.choice(sorted_pos_aa[:min(3, len(sorted_pos_aa))])
                        sequence += chosen_aa
                    else:
                        # 使用全局最佳氨基酸
                        weights = [aa_stats[aa]['mean_activity'] for aa in high_activity_aa]
                        chosen_aa = np.random.choice(high_activity_aa, p=np.array(weights)/sum(weights))
                        sequence += chosen_aa
            else:
                # 使用全局优化策略
                for _ in range(target_length):
                    weights = [aa_stats[aa]['mean_activity'] for aa in high_activity_aa]
                    chosen_aa = np.random.choice(high_activity_aa, p=np.array(weights)/sum(weights))
                    sequence += chosen_aa
            
            design3_sequences.append(sequence)
            print(f"  设计{i+1:2d}: {sequence}")
    
    # 合并所有设计
    all_designed_sequences = design1_sequences + design2_sequences + design3_sequences
    
    # 去重
    unique_sequences = list(set(all_designed_sequences))
    
    print(f"\n总设计序列数: {len(all_designed_sequences)}")
    print(f"去重后序列数: {len(unique_sequences)}")
    
    return unique_sequences

def main():
    """主函数"""
    print("=" * 50)
    print("基于活性位点分析的抗菌多肽理性设计")
    print("=" * 50)
    
    # 1. 加载数据
    df = load_and_analyze_data()
    
    # 2. 位置特异性分析
    position_contributions, optimal_length, optimal_sequence = position_specific_analysis(df)
    
    # 3. Motif分析
    motif_activities = motif_analysis(df)
    
    # 4. 氨基酸偏好性分析
    aa_stats = amino_acid_preference_analysis(df)
    
    # 5. 理性设计新序列
    designed_sequences = design_optimized_peptides(
        position_contributions, motif_activities, aa_stats, target_length=12
    )
    
    # 6. 保存设计结果
    design_df = pd.DataFrame({
        'sequence': designed_sequences,
        'design_method': ['rational_design'] * len(designed_sequences)
    })
    
    output_file = 'rationally_designed_peptides.csv'
    design_df.to_csv(output_file, index=False)
    
    print(f"\n设计的序列已保存到: {output_file}")
    print(f"可以使用预测脚本进行活性预测:")
    print(f"python src/predict_peptide.py")
    
    # 7. 生成分析报告
    report_file = 'rational_design_analysis_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("抗菌多肽理性设计分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. 数据概况\n")
        f.write(f"   总序列数: {len(df)}\n")
        f.write(f"   活性范围: {df['activity'].min():.3f} - {df['activity'].max():.3f}\n")
        f.write(f"   平均活性: {df['activity'].mean():.3f}\n\n")
        
        f.write("2. 氨基酸活性贡献 (Top 10)\n")
        sorted_aa = sorted(aa_stats.items(), key=lambda x: x[1]['mean_activity'], reverse=True)
        for i, (aa, stats) in enumerate(sorted_aa[:10]):
            f.write(f"   {i+1:2d}. {aa}: {stats['mean_activity']:.3f} (频率: {stats['frequency']:.3f})\n")
        f.write("\n")
        
        f.write("3. 设计序列统计\n")
        f.write(f"   总设计序列: {len(designed_sequences)}\n")
        f.write(f"   理论最优序列: {optimal_sequence}\n")
        f.write(f"   最优长度: {optimal_length}\n\n")
        
        f.write("4. 高活性motif (Top 10)\n")
        high_motifs = [(motif, np.mean(acts)) for motif, acts in motif_activities.items() 
                      if len(acts) >= 3]
        high_motifs.sort(key=lambda x: x[1], reverse=True)
        for i, (motif, avg_act) in enumerate(high_motifs[:10]):
            f.write(f"   {i+1:2d}. {motif}: {avg_act:.3f}\n")
    
    print(f"分析报告已保存到: {report_file}")
    
    return {
        'designed_sequences': designed_sequences,
        'position_contributions': position_contributions,
        'motif_activities': motif_activities,
        'aa_stats': aa_stats,
        'optimal_sequence': optimal_sequence
    }

if __name__ == '__main__':
    results = main()
