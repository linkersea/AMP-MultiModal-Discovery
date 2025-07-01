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
            import glob
            
            # 首先尝试加载最终综合结果文件（包含AI预测得分）
            final_files = glob.glob(f'{self.results_dir}/final_predicted_candidates*.csv')
            
            if final_files:
                print("🎯 发现最终综合结果文件，使用AI预测得分进行分析...")
                self.combined_df = pd.read_csv(final_files[0])
                
                # 分离各方法的数据
                self.seq_var_df = self.combined_df[self.combined_df['method'] == 'sequence_variation'].copy()
                self.rational_df = self.combined_df[self.combined_df['method'].str.contains('rational_design')].copy()
                self.vae_df = self.combined_df[self.combined_df['method'] == 'vae_generation'].copy()
                
                print(f"✅ 数据加载成功 (来源: {final_files[0]}):")
                print(f"  序列变异: {len(self.seq_var_df)} 序列")
                print(f"  理性设计: {len(self.rational_df)} 序列") 
                print(f"  VAE生成: {len(self.vae_df)} 序列")
                print(f"  总计: {len(self.combined_df)} 序列")
                
                # 检查AI预测得分
                if 'predicted_activity' in self.combined_df.columns:
                    ai_scores = self.combined_df['predicted_activity'].dropna()
                    print(f"📊 AI预测得分统计:")
                    print(f"  平均得分: {ai_scores.mean():.3f}")
                    print(f"  高分序列 (>0.8): {sum(ai_scores > 0.8)} 个")
                    print(f"  中分序列 (0.6-0.8): {sum((ai_scores >= 0.6) & (ai_scores <= 0.8))} 个")
                    print(f"  低分序列 (<0.6): {sum(ai_scores < 0.6)} 个")
                
            else:
                # 回退到原来的方法：分别加载三个子目录
                print("⚠️ 未找到最终综合文件，回退到分别加载子目录...")
                
                seq_var_files = glob.glob(f'{self.results_dir}/sequence_variation/*candidates*.csv')
                rational_files = glob.glob(f'{self.results_dir}/rational_design/*candidates*.csv')
                vae_files = glob.glob(f'{self.results_dir}/vae_generation/*candidates*.csv')
                
                if not seq_var_files:
                    raise FileNotFoundError(f"未找到序列变异结果文件: {self.results_dir}/sequence_variation/")
                if not rational_files:
                    raise FileNotFoundError(f"未找到理性设计结果文件: {self.results_dir}/rational_design/")
                if not vae_files:
                    raise FileNotFoundError(f"未找到VAE生成结果文件: {self.results_dir}/vae_generation/")
                
                # 加载数据
                self.seq_var_df = pd.read_csv(seq_var_files[0])
                self.rational_df = pd.read_csv(rational_files[0])
                self.vae_df = pd.read_csv(vae_files[0])
                
                # 添加方法标识列
                self.seq_var_df['method'] = 'sequence_variation'
                self.rational_df['method'] = 'rational_design'
                self.vae_df['method'] = 'vae_generation'
                
                # 合并所有数据
                self.combined_df = pd.concat([self.seq_var_df, self.rational_df, self.vae_df], ignore_index=True)
                
                print(f"✅ 数据加载成功:")
                print(f"  序列变异: {len(self.seq_var_df)} 序列 ({seq_var_files[0]})")
                print(f"  理性设计: {len(self.rational_df)} 序列 ({rational_files[0]})")
                print(f"  VAE生成: {len(self.vae_df)} 序列 ({vae_files[0]})")
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
        
        # 检查可用的评分列
        print("🔍 可用评分列:")
        available_cols = list(self.combined_df.columns)
        score_cols = [col for col in available_cols if any(keyword in col.lower() 
                     for keyword in ['score', 'activity', 'prediction', 'probability'])]
        print(f"   {score_cols}")
        
        # 优先使用AI预测得分
        if 'predicted_activity' in self.combined_df.columns:
            primary_score = 'predicted_activity'
            score_name = "AI预测活性"
            print(f"✅ 使用主要评分标准: {score_name}")
        elif 'biological_score' in self.combined_df.columns:
            primary_score = 'biological_score'
            score_name = "生物学评分"
            print(f"⚠️ 使用备用评分标准: {score_name}")
        else:
            primary_score = None
            score_name = "无统一评分"
            print(f"❌ 未找到统一评分标准")
        
        for method in ['sequence_variation', 'rational_design', 'vae_generation']:
            # 处理理性设计的方法名变体
            if method == 'rational_design':
                method_df = self.combined_df[self.combined_df['method'].str.contains('rational_design')]
                method_display = "理性设计"
            elif method == 'sequence_variation':
                method_df = self.combined_df[self.combined_df['method'] == method]
                method_display = "序列变异"
            elif method == 'vae_generation':
                method_df = self.combined_df[self.combined_df['method'] == method]
                method_display = "VAE生成"
            
            if len(method_df) == 0:
                print(f"\n🔸 {method_display}: 无数据")
                continue
                
            print(f"\n🔸 {method_display} Top {top_n}:")
            print(f"   总序列数: {len(method_df)}")
            
            if primary_score and primary_score in method_df.columns:
                # 使用主要评分排序
                valid_scores = method_df[method_df[primary_score].notna()]
                if len(valid_scores) > 0:
                    top_sequences = valid_scores.nlargest(top_n, primary_score)
                    print(f"   排序依据: {score_name}")
                else:
                    top_sequences = method_df.head(top_n)
                    print(f"   排序依据: 原始顺序 (无有效{score_name})")
            else:
                # 回退到其他评分标准
                if 'biological_score' in method_df.columns:
                    top_sequences = method_df.nlargest(top_n, 'biological_score')
                    print(f"   排序依据: 生物学评分")
                else:
                    top_sequences = method_df.head(top_n)
                    print(f"   排序依据: 原始顺序")
            
            # 显示Top序列
            for i, (_, row) in enumerate(top_sequences.iterrows(), 1):
                seq = row['sequence']
                
                # 收集所有可用得分
                scores_info = []
                if 'predicted_activity' in row and pd.notna(row['predicted_activity']):
                    scores_info.append(f"AI: {row['predicted_activity']:.3f}")
                if 'biological_score' in row and pd.notna(row['biological_score']):
                    scores_info.append(f"Bio: {row['biological_score']:.1f}")
                
                scores_str = ", ".join(scores_info) if scores_info else "无评分"
                print(f"  {i:2d}. {seq:<20} ({scores_str})")
        
        # 全局Top候选（跨方法）
        if primary_score and primary_score in self.combined_df.columns:
            print(f"\n🏆 全局Top {top_n} 候选序列 (基于{score_name}):")
            valid_global = self.combined_df[self.combined_df[primary_score].notna()]
            if len(valid_global) > 0:
                global_top = valid_global.nlargest(top_n, primary_score)
                for i, (_, row) in enumerate(global_top.iterrows(), 1):
                    seq = row['sequence']
                    method = row['method']
                    score = row[primary_score]
                    bio_score = row.get('biological_score', 'N/A')
                    print(f"  {i:2d}. {seq:<20} (AI: {score:.3f}, Bio: {bio_score}, 方法: {method})")
    
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
            
            # AI预测得分统计
            if 'predicted_activity' in self.combined_df.columns:
                ai_scores = self.combined_df['predicted_activity'].dropna()
                f.write("### 🤖 AI预测得分分布\n\n")
                f.write(f"- 平均得分: {ai_scores.mean():.3f}\n")
                f.write(f"- 高分序列 (>0.8): {sum(ai_scores > 0.8)} 个\n")
                f.write(f"- 中分序列 (0.6-0.8): {sum((ai_scores >= 0.6) & (ai_scores <= 0.8))} 个\n")
                f.write(f"- 低分序列 (<0.6): {sum(ai_scores < 0.6)} 个\n\n")
            
            # Top候选序列 - 优先使用AI预测得分
            f.write("## 🎯 推荐实验候选序列\n\n")
            f.write("### 第一优先级 (立即合成验证)\n\n")
            
            # 全局Top 15（基于AI预测得分）
            if 'predicted_activity' in self.combined_df.columns:
                valid_ai = self.combined_df[self.combined_df['predicted_activity'].notna()]
                if len(valid_ai) > 0:
                    global_top15 = valid_ai.nlargest(15, 'predicted_activity')
                    f.write("#### 全局Top 15 (基于AI预测活性):\n\n")
                    for i, (_, row) in enumerate(global_top15.iterrows(), 1):
                        seq = row['sequence']
                        ai_score = row['predicted_activity']
                        bio_score = row.get('biological_score', 'N/A')
                        method = row['method']
                        f.write(f"{i}. `{seq}` (AI: {ai_score:.3f}, Bio: {bio_score}, 来源: {method})\n")
                    f.write("\n")
            
            # 分方法Top 5
            for method, method_name in [('sequence_variation', '序列变异'), 
                                      ('rational_design', '理性设计'), 
                                      ('vae_generation', 'VAE生成')]:
                if method == 'rational_design':
                    method_df = self.combined_df[self.combined_df['method'].str.contains('rational_design')]
                else:
                    method_df = self.combined_df[self.combined_df['method'] == method]
                
                if len(method_df) == 0:
                    continue
                    
                f.write(f"\n#### {method_name} Top 5:\n")
                
                # 优先使用AI预测得分
                if 'predicted_activity' in method_df.columns:
                    valid_scores = method_df[method_df['predicted_activity'].notna()]
                    if len(valid_scores) > 0:
                        top_5 = valid_scores.nlargest(5, 'predicted_activity')
                    else:
                        top_5 = method_df.head(5)
                elif 'biological_score' in method_df.columns:
                    top_5 = method_df.nlargest(5, 'biological_score')
                else:
                    top_5 = method_df.head(5)
                
                for i, (_, row) in enumerate(top_5.iterrows(), 1):
                    seq = row['sequence']
                    ai_score = row.get('predicted_activity', 'N/A')
                    bio_score = row.get('biological_score', 'N/A')
                    f.write(f"{i}. `{seq}` (AI: {ai_score}, Bio: {bio_score})\n")
            
            # 分析建议
            f.write("\n## 💡 分析建议\n\n")
            f.write("### 实验设计建议\n")
            f.write("1. **第一批实验**: 选择全局Top 15序列进行合成验证\n")
            f.write("2. **对照组**: 包含2-3个已知活性序列\n") 
            f.write("3. **浓度范围**: 建议测试1-128 μg/mL\n")
            f.write("4. **细菌株**: 包含革兰氏阳性和阴性菌\n\n")
            
            f.write("### 评分标准说明\n")
            f.write("- **AI预测活性**: 基于多模态深度学习模型的预测得分 (0-1)\n")
            f.write("- **生物学评分**: 基于理化性质和生物学知识的评分 (0-100)\n")
            f.write("- **建议优先级**: AI预测活性 > 生物学评分 > 序列特征\n\n")
            
            f.write("### 后续优化方向\n")
            f.write("- 根据实验结果调整AI模型参数\n")
            f.write("- 分析成功序列的共同特征\n")
            f.write("- 收集实验反馈进行模型微调\n")
            f.write("- 扩展到更多细菌株测试\n")
        
        print(f"\n✅ 详细分析报告已生成: {report_file}")
    
    def add_unified_ai_scores(self):
        """为所有序列添加统一的AI预测得分"""
        print("\n🤖 正在为所有序列计算AI预测得分...")
        
        try:
            # 导入预测模块
            import sys
            sys.path.append('src')
            from predict_peptide import PeptidePredictionPipeline
            
            # 初始化预测器
            predictor = PeptidePredictionPipeline()
            
            # 为合并数据集添加AI得分
            sequences = self.combined_df['sequence'].tolist()
            ai_scores = []
            
            for i, seq in enumerate(sequences):
                if i % 10 == 0:
                    print(f"   进度: {i+1}/{len(sequences)}")
                
                try:
                    score = predictor.predict_single(seq)
                    ai_scores.append(score)
                except Exception as e:
                    print(f"   警告: 序列 {seq} 预测失败: {e}")
                    ai_scores.append(0.0)  # 默认低分
            
            # 添加AI得分列
            self.combined_df['ai_prediction_score'] = ai_scores
            
            # 为各个子数据集也添加AI得分
            seq_var_sequences = self.seq_var_df['sequence'].tolist()
            rational_sequences = self.rational_df['sequence'].tolist()
            vae_sequences = self.vae_df['sequence'].tolist()
            
            # 从合并数据集中提取对应得分
            self.seq_var_df['ai_prediction_score'] = [
                self.combined_df[self.combined_df['sequence'] == seq]['ai_prediction_score'].iloc[0]
                for seq in seq_var_sequences
            ]
            
            self.rational_df['ai_prediction_score'] = [
                self.combined_df[self.combined_df['sequence'] == seq]['ai_prediction_score'].iloc[0]
                for seq in rational_sequences
            ]
            
            self.vae_df['ai_prediction_score'] = [
                self.combined_df[self.combined_df['sequence'] == seq]['ai_prediction_score'].iloc[0]
                for seq in vae_sequences
            ]
            
            print(f"✅ AI评分完成! 平均得分: {np.mean(ai_scores):.3f}")
            print(f"   高分序列 (>0.8): {sum(1 for s in ai_scores if s > 0.8)} 个")
            print(f"   中分序列 (0.6-0.8): {sum(1 for s in ai_scores if 0.6 <= s <= 0.8)} 个")
            print(f"   低分序列 (<0.6): {sum(1 for s in ai_scores if s < 0.6)} 个")
            
        except Exception as e:
            print(f"⚠️ AI评分失败: {e}")
            print("   将使用现有评分进行分析")
            return False
        
        return True
    
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
    import argparse
    import glob
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='多肽发现结果分析工具')
    parser.add_argument('--results_dir', type=str, help='指定结果目录路径')
    parser.add_argument('--include_plots', action='store_true', help='生成可视化图表')
    parser.add_argument('--plot_format', default='png', choices=['png', 'pdf', 'svg'], help='图表格式')
    args = parser.parse_args()
    
    # 确定分析目录
    if args.results_dir:
        if not os.path.exists(args.results_dir):
            print(f"❌ 指定的结果目录不存在: {args.results_dir}")
            return
        latest_dir = args.results_dir
        print(f"🔍 分析指定目录: {latest_dir}")
    else:
        # 使用最新的结果目录
        result_dirs = glob.glob('results_three_methods_*')
        if not result_dirs:
            print("❌ 未找到结果目录！请先运行 three_method_discovery.py")
            return
        latest_dir = sorted(result_dirs)[-1]  # 选择最新的结果
        print(f"🔍 分析最新目录: {latest_dir}")
    
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
