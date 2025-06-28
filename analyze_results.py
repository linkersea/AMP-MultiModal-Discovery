#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多肽发现结果快速分析工具
一键生成全面的结果分析报告
"""

import pandas as pd
import numpy as np
from collections import Counter
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class PeptideResultAnalyzer:
    """多肽结果分析器"""
    
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.seq_var_df = None
        self.rational_df = None
        self.vae_df = None
        self.combined_df = None
        
    def load_data(self):
        """加载三种方法的结果数据"""
        try:
            self.seq_var_df = pd.read_csv(f'{self.results_dir}/sequence_variation/candidates.csv')
            self.rational_df = pd.read_csv(f'{self.results_dir}/rational_design/candidates.csv')
            self.vae_df = pd.read_csv(f'{self.results_dir}/vae_generation/candidates.csv')
            
            # 合并所有数据
            self.combined_df = pd.concat([self.seq_var_df, self.rational_df, self.vae_df], ignore_index=True)
            
            print(f"✅ 数据加载成功:")
            print(f"  序列变异: {len(self.seq_var_df)} 序列")
            print(f"  理性设计: {len(self.rational_df)} 序列")
            print(f"  VAE生成: {len(self.vae_df)} 序列")
            print(f"  总计: {len(self.combined_df)} 序列")
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
        return True
    
    def analyze_amino_acid_usage(self):
        """分析氨基酸使用模式"""
        print("\n" + "="*60)
        print("氨基酸使用模式分析")
        print("="*60)
        
        def get_aa_frequency(df, method_name):
            all_sequences = ''.join(df['sequence'].tolist())
            aa_counts = Counter(all_sequences)
            total_aas = len(all_sequences)
            
            print(f"\n🔸 {method_name}")
            print(f"总氨基酸数: {total_aas}")
            print("Top 8 氨基酸频率:")
            
            for aa, count in aa_counts.most_common(8):
                percentage = count / total_aas * 100
                print(f"  {aa}: {percentage:5.1f}% ({count:3d})")
            
            return aa_counts
        
        seq_var_freq = get_aa_frequency(self.seq_var_df, "序列变异")
        rational_freq = get_aa_frequency(self.rational_df, "理性设计")
        vae_freq = get_aa_frequency(self.vae_df, "VAE生成")
        
        return seq_var_freq, rational_freq, vae_freq
    
    def analyze_sequence_properties(self):
        """分析序列物理化学性质"""
        print("\n" + "="*60)
        print("序列物理化学性质分析")
        print("="*60)
        
        def calculate_properties(df, method_name):
            lengths = df['sequence'].apply(len)
            charges = df['sequence'].apply(self._calculate_net_charge)
            hydrophobic_ratios = df['sequence'].apply(self._calculate_hydrophobic_ratio)
            aromatic_ratios = df['sequence'].apply(self._calculate_aromatic_ratio)
            
            print(f"\n🔸 {method_name}")
            print(f"长度分布: {lengths.min()}-{lengths.max()}, 平均: {lengths.mean():.1f}")
            print(f"净电荷: {charges.min()}-{charges.max()}, 平均: {charges.mean():.1f}")
            print(f"疏水性比例: {hydrophobic_ratios.min():.2f}-{hydrophobic_ratios.max():.2f}, 平均: {hydrophobic_ratios.mean():.2f}")
            print(f"芳香性比例: {aromatic_ratios.min():.2f}-{aromatic_ratios.max():.2f}, 平均: {aromatic_ratios.mean():.2f}")
            
            return {
                'lengths': lengths,
                'charges': charges, 
                'hydrophobic_ratios': hydrophobic_ratios,
                'aromatic_ratios': aromatic_ratios
            }
        
        seq_var_props = calculate_properties(self.seq_var_df, "序列变异")
        rational_props = calculate_properties(self.rational_df, "理性设计")
        vae_props = calculate_properties(self.vae_df, "VAE生成")
        
        return seq_var_props, rational_props, vae_props
    
    def find_top_candidates(self, top_n=10):
        """寻找各方法的顶级候选序列"""
        print("\n" + "="*60)
        print(f"Top {top_n} 候选序列分析")
        print("="*60)
        
        # 确保有生物学评分列
        if 'biological_score' not in self.combined_df.columns:
            print("⚠️ 缺少biological_score列，使用长度作为排序依据")
            score_col = 'length'
        else:
            score_col = 'biological_score'
        
        for method in ['sequence_variation', 'rational_design', 'vae_generation']:
            method_df = self.combined_df[self.combined_df['method'] == method]
            if score_col in method_df.columns:
                top_sequences = method_df.nlargest(top_n, score_col)
            else:
                top_sequences = method_df.head(top_n)
                
            print(f"\n🔸 {method} Top {top_n}:")
            for i, (_, row) in enumerate(top_sequences.iterrows(), 1):
                seq = row['sequence']
                score = row.get(score_col, 'N/A')
                print(f"  {i:2d}. {seq:<20} (评分: {score})")
    
    def analyze_sequence_diversity(self):
        """分析序列多样性"""
        print("\n" + "="*60)
        print("序列多样性分析")
        print("="*60)
        
        def calculate_diversity_metrics(sequences):
            # 计算序列长度多样性
            lengths = [len(seq) for seq in sequences]
            length_diversity = len(set(lengths))
            
            # 计算氨基酸组成多样性
            all_aas = ''.join(sequences)
            aa_types = len(set(all_aas))
            
            # 计算序列独特性
            unique_sequences = len(set(sequences))
            total_sequences = len(sequences)
            uniqueness = unique_sequences / total_sequences
            
            return {
                'length_diversity': length_diversity,
                'aa_types': aa_types,
                'uniqueness': uniqueness,
                'unique_count': unique_sequences,
                'total_count': total_sequences
            }
        
        # 分析每种方法的多样性
        methods = [
            ('序列变异', self.seq_var_df['sequence'].tolist()),
            ('理性设计', self.rational_df['sequence'].tolist()),
            ('VAE生成', self.vae_df['sequence'].tolist()),
            ('全部方法', self.combined_df['sequence'].tolist())
        ]
        
        for method_name, sequences in methods:
            metrics = calculate_diversity_metrics(sequences)
            print(f"\n🔸 {method_name}:")
            print(f"  长度多样性: {metrics['length_diversity']} 种不同长度")
            print(f"  氨基酸种类: {metrics['aa_types']} 种")
            print(f"  序列独特性: {metrics['uniqueness']:.1%} ({metrics['unique_count']}/{metrics['total_count']})")
    
    def generate_analysis_report(self):
        """生成完整分析报告"""
        report_file = f"{self.results_dir}/detailed_analysis_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# 多肽发现结果详细分析报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
            f.write(f"**结果目录**: {self.results_dir}\n\n")
            
            # 基本统计
            f.write("## 📊 基本统计\n\n")
            f.write(f"- 序列变异: {len(self.seq_var_df)} 个序列\n")
            f.write(f"- 理性设计: {len(self.rational_df)} 个序列\n")
            f.write(f"- VAE生成: {len(self.vae_df)} 个序列\n")
            f.write(f"- **总计**: {len(self.combined_df)} 个候选序列\n\n")
            
            # Top候选序列
            f.write("## 🎯 推荐实验候选序列\n\n")
            f.write("### 第一优先级 (立即合成验证)\n")
            
            # 从每种方法选择top 5
            for method, method_name in [('sequence_variation', '序列变异'), 
                                      ('rational_design', '理性设计'), 
                                      ('vae_generation', 'VAE生成')]:
                method_df = self.combined_df[self.combined_df['method'] == method]
                if 'biological_score' in method_df.columns:
                    top_5 = method_df.nlargest(5, 'biological_score')
                else:
                    top_5 = method_df.head(5)
                
                f.write(f"\n#### {method_name} Top 5:\n")
                for i, (_, row) in enumerate(top_5.iterrows(), 1):
                    seq = row['sequence']
                    score = row.get('biological_score', 'N/A')
                    f.write(f"{i}. `{seq}` (评分: {score})\n")
            
            # 分析建议
            f.write("\n## 💡 分析建议\n\n")
            f.write("### 实验设计建议\n")
            f.write("1. **第一批实验**: 选择上述Top 15个序列 (每种方法5个)\n")
            f.write("2. **对照组**: 包含2-3个已知活性序列\n") 
            f.write("3. **浓度范围**: 建议测试1-128 μg/mL\n")
            f.write("4. **细菌株**: 包含革兰氏阳性和阴性菌\n\n")
            
            f.write("### 后续优化方向\n")
            f.write("- 根据实验结果调整生物学评分函数\n")
            f.write("- 分析成功序列的共同特征\n")
            f.write("- 优化VAE模型参数\n")
            f.write("- 扩展到更多细菌株测试\n")
        
        print(f"\n✅ 详细分析报告已生成: {report_file}")
    
    def _calculate_net_charge(self, sequence):
        """计算净电荷"""
        positive = sequence.count('K') + sequence.count('R') + sequence.count('H')
        negative = sequence.count('D') + sequence.count('E')
        return positive - negative
    
    def _calculate_hydrophobic_ratio(self, sequence):
        """计算疏水性氨基酸比例"""
        hydrophobic = sum(1 for aa in sequence if aa in 'AILMFWYV')
        return hydrophobic / len(sequence)
    
    def _calculate_aromatic_ratio(self, sequence):
        """计算芳香族氨基酸比例"""
        aromatic = sum(1 for aa in sequence if aa in 'FWY')
        return aromatic / len(sequence)

def main():
    """主分析函数"""
    # 使用最新的结果目录
    import glob
    result_dirs = glob.glob('results_three_methods_*')
    if not result_dirs:
        print("❌ 未找到结果目录！请先运行 three_method_discovery.py")
        return
    
    latest_dir = sorted(result_dirs)[-1]  # 选择最新的结果
    print(f"🔍 分析目录: {latest_dir}")
    
    # 创建分析器并运行分析
    analyzer = PeptideResultAnalyzer(latest_dir)
    
    if not analyzer.load_data():
        return
    
    # 执行各项分析
    analyzer.analyze_amino_acid_usage()
    analyzer.analyze_sequence_properties()
    analyzer.find_top_candidates(top_n=10)
    analyzer.analyze_sequence_diversity()
    analyzer.generate_analysis_report()
    
    print("\n" + "="*60)
    print("🎉 分析完成！")
    print("="*60)
    print("📋 查看详细报告:")
    print(f"   {latest_dir}/detailed_analysis_report.md")
    print("\n💡 下一步建议:")
    print("1. 根据报告选择实验候选序列")
    print("2. 设计合成和活性测试实验")
    print("3. 收集实验反馈优化模型")

if __name__ == "__main__":
    main()
