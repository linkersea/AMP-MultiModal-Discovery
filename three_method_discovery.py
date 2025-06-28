#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三方法集成多肽发现框架
序列变异 + 理性设计 + VAE生成的完整解决方案
集成了外部模型进行最终筛选和排序
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
import sys
import argparse
sys.path.append('src')

# 导入各个方法模块
try:
    from advanced_vae_generator import VAEPeptideGenerator
except ImportError:
    print("警告: VAE模块导入失败，将使用简化版本")
    VAEPeptideGenerator = None

# 导入AI预测模型
try:
    from predict_peptide import PhysChemSeqEngBioBERTPredictor
except ImportError:
    print("警告: AI预测模块导入失败，将无法使用AI模型进行最终筛选")
    PhysChemSeqEngBioBERTPredictor = None

class ThreeMethodPeptideDiscovery:
    """三方法集成多肽发现系统"""
    
    def __init__(self, timestamp=None, model_path=None, scaler_path=None):
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"results_three_methods_{self.timestamp}"
        self.all_candidates = []
        self.method_results = {}
        self.predictor = None
        
        # 创建结果目录
        self._setup_directories()
        
        # 加载AI预测模型
        if PhysChemSeqEngBioBERTPredictor and model_path and os.path.exists(model_path):
            try:
                print("=" * 60)
                print("正在加载多模态深度学习CNN分类器...")
                self.predictor = PhysChemSeqEngBioBERTPredictor(model_path, scaler_path)
                print("✅ 多模态深度学习CNN分类器加载成功！")
                print("=" * 60)
            except Exception as e:
                print(f"⚠️ 警告: AI预测模型加载失败: {e}。")
                print("将回退到基于规则的生物学评分。")
                print("=" * 60)
                self.predictor = None
        else:
            print("=" * 60)
            print("⚠️ 警告: 未提供分类器路径或模型不存在，将使用基于规则的评分。")
            print("=" * 60)

    def _setup_directories(self):
        """设置结果目录结构"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        methods = ['sequence_variation', 'rational_design', 'vae_generation']
        for method in methods:
            method_dir = os.path.join(self.results_dir, method)
            if not os.path.exists(method_dir):
                os.makedirs(method_dir)
    
    def run_sequence_variation(self, target_count=200):
        """方法1: 序列变异 - 传统空间探索"""
        print("=" * 60)
        print("方法1: 序列变异 - 传统空间探索")
        print("=" * 60)
        
        method_dir = os.path.join(self.results_dir, 'sequence_variation')
        
        # 加载原始数据
        df = pd.read_csv('data/raw/120dataset.csv')
        high_activity = df[df['activity'] >= df['activity'].quantile(0.75)]
        
        # 记录探索参数
        exploration_log = {
            'method': 'sequence_variation',
            'strategy': 'local_neighborhood_search',
            'timestamp': self.timestamp,
            'source_sequences': len(df),
            'high_activity_seeds': len(high_activity),
            'target_count': target_count,
            'mutation_parameters': {
                'types': ['substitute', 'insert', 'delete'],
                'max_mutations_per_sequence': 3,
                'length_range': [8, 16]
            }
        }
        
        candidates = []
        seed_sequences = high_activity['sequence'].tolist()
        
        print(f"基于 {len(seed_sequences)} 个高活性种子序列生成变异体...")
        
        for i in range(target_count):
            if seed_sequences:
                base_seq = np.random.choice(seed_sequences)
                
                # 执行1-3个变异
                num_mutations = np.random.randint(1, 4)
                mutated_seq = self._mutate_sequence(base_seq, num_mutations)
                
                # 基本质量检查
                if 8 <= len(mutated_seq) <= 16 and self._is_valid_sequence(mutated_seq):
                    candidates.append({
                        'sequence': mutated_seq,
                        'method': 'sequence_variation',
                        'base_sequence': base_seq,
                        'num_mutations': num_mutations,
                        'final_length': len(mutated_seq),
                        'exploration_strategy': 'local_search'
                    })
        
        # 去重
        unique_candidates = self._remove_duplicates(candidates)
        
        # 保存结果
        self._save_method_results('sequence_variation', unique_candidates, exploration_log)
        
        print(f"✅ 序列变异完成: 生成 {len(unique_candidates)} 个独特候选序列")
        self.method_results['sequence_variation'] = unique_candidates
        
        return unique_candidates
    
    def run_rational_design(self, target_count=100):
        """方法2: 理性设计 - 知识驱动设计"""
        print("=" * 60)
        print("方法2: 理性设计 - 知识驱动设计")
        print("=" * 60)
        
        method_dir = os.path.join(self.results_dir, 'rational_design')
        
        # 设计参数
        design_parameters = {
            'length_range': [10, 14],
            'net_charge_range': [2, 6],
            'hydrophobic_ratio_range': [0.3, 0.6],
            'aromatic_ratio_range': [0.1, 0.3],
            'key_amino_acids': ['R', 'K', 'W', 'F'],
            'design_patterns': [
                'amphipathic_helix',
                'beta_sheet',
                'random_coil'
            ]
        }
        
        exploration_log = {
            'method': 'rational_design',
            'strategy': 'knowledge_driven_design',
            'timestamp': self.timestamp,
            'design_parameters': design_parameters,
            'target_count': target_count
        }
        
        candidates = []
        
        print("基于抗菌肽设计原理生成候选序列...")
        
        for i in range(target_count):
            # 随机选择设计模式
            pattern = np.random.choice(design_parameters['design_patterns'])
            
            if pattern == 'amphipathic_helix':
                sequence = self._design_amphipathic_helix()
            elif pattern == 'beta_sheet':
                sequence = self._design_beta_sheet()
            else:  # random_coil
                sequence = self._design_random_coil()
            
            if sequence and self._is_valid_sequence(sequence):
                bio_score = self._calculate_biological_score(sequence)
                
                candidates.append({
                    'sequence': sequence,
                    'method': 'rational_design',
                    'design_pattern': pattern,
                    'biological_score': bio_score,
                    'length': len(sequence),
                    'exploration_strategy': 'knowledge_guided'
                })
        
        # 按生物学评分排序并选择最佳候选
        candidates.sort(key=lambda x: x['biological_score'], reverse=True)
        unique_candidates = self._remove_duplicates(candidates)
        
        # 保存结果
        self._save_method_results('rational_design', unique_candidates, exploration_log)
        
        print(f"✅ 理性设计完成: 生成 {len(unique_candidates)} 个设计序列")
        self.method_results['rational_design'] = unique_candidates
        
        return unique_candidates
    
    def run_vae_generation(self, target_count=150, ai_score_threshold=0.5):
        """方法3: VAE生成 - AI驱动创新"""
        print("=" * 60)
        print("方法3: VAE生成 - AI驱动创新")
        if self.predictor:
            print("模式: AI分类器引导生成")
        else:
            print("模式: 启发式规则引导生成")
        print("=" * 60)
        
        method_dir = os.path.join(self.results_dir, 'vae_generation')
        
        exploration_log = {
            'method': 'vae_generation',
            'strategy': 'ai_driven_innovation' if self.predictor else 'heuristic_fallback',
            'timestamp': self.timestamp,
            'target_count': target_count,
            'model_parameters': {
                'latent_dimension': 16,
                'training_epochs': 50,
                'temperature_sampling': 1.0,
                'feedback_enabled': True,
                'ai_predictor_integrated': bool(self.predictor),
                'ai_score_threshold': ai_score_threshold if self.predictor else 'N/A'
            }
        }
        
        candidates = []
        
        try:
            if VAEPeptideGenerator is not None:
                print("使用真正的VAE模型进行生成...")
                
                # 将AI预测器注入VAE生成器
                vae_generator = VAEPeptideGenerator(predictor=self.predictor)
                
                # 快速训练模式
                print("准备训练数据...")
                vae_generator.prepare_data()
                
                print("构建并训练VAE模型...")
                vae_generator.build_model(latent_dim=16)
                vae_generator.train_model(epochs=30, batch_size=8)  # 快速训练
                
                print("生成新序列...")
                generated = vae_generator.generate_peptides_with_feedback(
                    num_samples=target_count,
                    temperature=1.0,
                    score_threshold=ai_score_threshold
                )
                
                # 转换格式
                for item in generated:
                    candidates.append({
                        'sequence': item['sequence'],
                        'method': 'vae_generation',
                        'predicted_activity': item.get('predicted_activity', 0),
                        'generation_strategy': 'ai_guided_latent_space_sampling' if self.predictor else 'heuristic_guided_sampling'
                    })
                
                exploration_log['vae_training_successful'] = True
                
            else:
                print("VAE模块不可用，使用高级启发式生成...")
                candidates = self._advanced_heuristic_generation(target_count)
                exploration_log['vae_training_successful'] = False
        
        except Exception as e:
            print(f"VAE训练失败: {e}")
            print("回退到高级启发式生成...")
            candidates = self._advanced_heuristic_generation(target_count)
            exploration_log['vae_training_successful'] = False
            exploration_log['error'] = str(e)
        
        # 去重和质量筛选
        unique_candidates = self._remove_duplicates(candidates)
        
        # 保存结果
        self._save_method_results('vae_generation', unique_candidates, exploration_log)
        
        print(f"✅ VAE生成完成: 生成 {len(unique_candidates)} 个创新序列")
        self.method_results['vae_generation'] = unique_candidates
        
        return unique_candidates
    
    def _predict_activity_for_all_candidates(self, candidates):
        """使用加载的AI模型为所有候选序列预测活性"""
        if not self.predictor or not candidates:
            print("AI预测器不可用或没有候选序列，跳过最终预测。")
            return candidates
        
        print("=" * 60)
        print("最终筛选: 使用多模态深度学习CNN分类器对所有候选序列进行活性预测...")
        print("=" * 60)
        
        sequences = [c['sequence'] for c in candidates]
        
        try:
            y_prob, _ = self.predictor.predict(sequences)
            
            # 更新每个候选序列的预测活性
            for i, candidate in enumerate(candidates):
                # 如果已有预测值（来自VAE），则保留；否则更新
                candidate['predicted_activity'] = round(float(y_prob[i]), 4)
            
            print(f"✅ 已为 {len(candidates)} 个序列更新AI预测活性。")
            
        except Exception as e:
            print(f"⚠️ AI预测过程中发生错误: {e}。无法更新预测分数。")
            
        return candidates

    def _mutate_sequence(self, sequence, num_mutations):
        """序列变异操作"""
        seq_list = list(sequence)
        amino_acids = ['R', 'W', 'K', 'I', 'V', 'F', 'Y', 'L', 'A']
        
        for _ in range(num_mutations):
            mutation_type = np.random.choice(['substitute', 'insert', 'delete'])
            
            if mutation_type == 'substitute' and len(seq_list) > 0:
                pos = np.random.randint(0, len(seq_list))
                seq_list[pos] = np.random.choice(amino_acids)
                
            elif mutation_type == 'insert' and len(seq_list) < 16:
                pos = np.random.randint(0, len(seq_list) + 1)
                seq_list.insert(pos, np.random.choice(amino_acids))
                
            elif mutation_type == 'delete' and len(seq_list) > 8:
                pos = np.random.randint(0, len(seq_list))
                seq_list.pop(pos)
        
        return ''.join(seq_list)
    
    def _design_amphipathic_helix(self):
        """设计两亲性螺旋结构"""
        # 交替放置疏水性和亲水性氨基酸
        hydrophobic = ['I', 'L', 'V', 'F', 'W', 'Y']
        hydrophilic = ['R', 'K', 'H', 'S', 'T', 'N', 'Q']
        
        length = np.random.randint(10, 15)
        sequence = ""
        
        for i in range(length):
            if i % 2 == 0:  # 疏水性位置
                sequence += np.random.choice(hydrophobic)
            else:  # 亲水性位置
                sequence += np.random.choice(hydrophilic)
        
        return sequence
    
    def _design_beta_sheet(self):
        """设计β折叠结构"""
        # β折叠倾向的氨基酸
        beta_preferred = ['I', 'V', 'F', 'Y', 'W', 'K', 'R']
        
        length = np.random.randint(10, 14)
        sequence = ""
        
        for i in range(length):
            sequence += np.random.choice(beta_preferred)
        
        # 确保有足够的正电荷
        positive_count = sequence.count('R') + sequence.count('K')
        if positive_count < 2:
            # 随机替换一些位置为正电荷氨基酸
            seq_list = list(sequence)
            for _ in range(2 - positive_count):
                pos = np.random.randint(0, len(seq_list))
                seq_list[pos] = np.random.choice(['R', 'K'])
            sequence = ''.join(seq_list)
        
        return sequence
    
    def _design_random_coil(self):
        """设计无规卷曲结构"""
        # 平衡的氨基酸组成
        amino_acids = ['R', 'K', 'W', 'F', 'I', 'V', 'L', 'A', 'G', 'Y']
        weights = [0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]
        
        length = np.random.randint(10, 14)
        sequence = ""
        
        for i in range(length):
            sequence += np.random.choice(amino_acids, p=weights)
        
        return sequence
    
    def _advanced_heuristic_generation(self, target_count):
        """高级启发式生成（VAE的替代方案）"""
        print("使用高级启发式方法模拟VAE生成...")
        
        candidates = []
        
        # 加载已知序列作为参考
        df = pd.read_csv('data/raw/120dataset.csv')
        reference_sequences = df['sequence'].tolist()
        
        for i in range(target_count):
            # 策略1: 组合已知模式 (30%)
            if np.random.random() < 0.3:
                sequence = self._combine_sequence_patterns(reference_sequences)
                strategy = 'pattern_combination'
            
            # 策略2: 模拟潜在空间插值 (40%)
            elif np.random.random() < 0.7:
                sequence = self._simulate_latent_interpolation(reference_sequences)
                strategy = 'simulated_interpolation'
            
            # 策略3: 随机创新生成 (30%)
            else:
                sequence = self._random_innovative_generation()
                strategy = 'random_innovation'
            
            if sequence and self._is_valid_sequence(sequence):
                candidates.append({
                    'sequence': sequence,
                    'method': 'vae_generation',
                    'generation_strategy': strategy,
                    'predicted_activity': self._calculate_biological_score(sequence) / 100,
                    'exploration_strategy': 'heuristic_simulation'
                })
        
        return candidates
    
    def _combine_sequence_patterns(self, reference_sequences):
        """组合已知序列模式"""
        # 随机选择2-3个参考序列
        selected = np.random.choice(reference_sequences, size=min(3, len(reference_sequences)), replace=False)
        
        # 提取片段并组合
        fragments = []
        for seq in selected:
            start = np.random.randint(0, max(1, len(seq) - 3))
            end = start + np.random.randint(3, min(6, len(seq) - start + 1))
            fragments.append(seq[start:end])
        
        # 组合片段
        combined = ''.join(fragments)
        
        # 调整长度
        if len(combined) > 16:
            combined = combined[:16]
        elif len(combined) < 8:
            # 填充到最小长度
            amino_acids = ['R', 'K', 'W', 'F', 'I', 'V']
            while len(combined) < 8:
                combined += np.random.choice(amino_acids)
        
        return combined
    
    def _simulate_latent_interpolation(self, reference_sequences):
        """模拟潜在空间插值"""
        # 选择两个参考序列
        seq1, seq2 = np.random.choice(reference_sequences, size=2, replace=False)
        
        # 计算"插值"序列
        max_len = max(len(seq1), len(seq2))
        min_len = min(len(seq1), len(seq2))
        
        # 随机选择插值比例
        alpha = np.random.random()
        
        # 生成插值长度
        interp_length = int(min_len + alpha * (max_len - min_len))
        interp_length = max(8, min(16, interp_length))
        
        # 构建插值序列
        sequence = ""
        for i in range(interp_length):
            # 从两个序列中选择氨基酸
            pos1 = min(i, len(seq1) - 1)
            pos2 = min(i, len(seq2) - 1)
            
            if np.random.random() < alpha:
                if pos1 < len(seq1):
                    sequence += seq1[pos1]
                else:
                    sequence += seq2[pos2]
            else:
                if pos2 < len(seq2):
                    sequence += seq2[pos2]
                else:
                    sequence += seq1[pos1]
        
        return sequence
    
    def _random_innovative_generation(self):
        """随机创新生成"""
        # 完全随机生成，但遵循抗菌肽的基本规律
        amino_acids = ['R', 'K', 'H', 'W', 'F', 'Y', 'I', 'V', 'L', 'A', 'G']
        weights = [0.25, 0.2, 0.05, 0.15, 0.1, 0.05, 0.08, 0.05, 0.03, 0.02, 0.02]
        
        length = np.random.randint(10, 15)
        sequence = ""
        
        for i in range(length):
            sequence += np.random.choice(amino_acids, p=weights)
        
        return sequence
    
    def _calculate_biological_score(self, sequence):
        """计算生物学评分"""
        score = 0
        length = len(sequence)
        
        # 净电荷
        positive = sequence.count('R') + sequence.count('K') + sequence.count('H')
        negative = sequence.count('D') + sequence.count('E')
        net_charge = positive - negative
        
        if 2 <= net_charge <= 6:
            score += 25
        elif 1 <= net_charge <= 8:
            score += 15
        
        # 疏水性
        hydrophobic = sum(sequence.count(aa) for aa in 'ILMFWYV')
        hydrophobic_ratio = hydrophobic / length
        
        if 0.3 <= hydrophobic_ratio <= 0.6:
            score += 20
        elif 0.2 <= hydrophobic_ratio <= 0.7:
            score += 15
        
        # 芳香族氨基酸
        aromatic = sequence.count('F') + sequence.count('W') + sequence.count('Y')
        if aromatic >= 1:
            score += 15
        
        # 长度优化
        if 10 <= length <= 14:
            score += 15
        elif 8 <= length <= 16:
            score += 10
        
        # 关键氨基酸
        if 'R' in sequence:
            score += 10
        if 'W' in sequence:
            score += 8
        
        return score
    
    def _is_valid_sequence(self, sequence):
        """检查序列有效性"""
        if not sequence or len(sequence) < 6 or len(sequence) > 20:
            return False
        
        # 检查是否只包含标准氨基酸
        standard_aa = set('ACDEFGHIKLMNPQRSTVWY')
        if not all(aa in standard_aa for aa in sequence.upper()):
            return False
        
        # 检查多样性
        if len(set(sequence)) < 3:
            return False
            
        return True

    def run_discovery_pipeline(self, sv_count=200, rd_count=100, vae_count=150):
        """运行完整的多肽发现流程"""
        print("🚀 开始三方法集成多肽发现流程...")
        
        # 运行各个生成方法
        sv_candidates = self.run_sequence_variation(target_count=sv_count)
        rd_candidates = self.run_rational_design(target_count=rd_count)
        vae_candidates = self.run_vae_generation(target_count=vae_count)
        
        # 合并所有候选序列
        self.all_candidates.extend(sv_candidates)
        self.all_candidates.extend(rd_candidates)
        self.all_candidates.extend(vae_candidates)
        
        # 使用AI模型对所有候选序列进行最终预测和评分
        self.all_candidates = self._predict_activity_for_all_candidates(self.all_candidates)
        
        # 保存最终结果
        self._save_final_results()
        
        # 生成综合报告
        self._generate_comprehensive_report()
        
        print("=" * 60)
        print("🎉 多肽发现流程全部完成！")
        print(f"最终结果保存在: {self.results_dir}")
        print("=" * 60)

    def _save_final_results(self):
        """保存最终候选结果"""
        final_df = pd.DataFrame(self.all_candidates)
        
        # 确保列存在
        if 'biological_score' not in final_df.columns:
            final_df['biological_score'] = final_df['sequence'].apply(self._calculate_biological_score)
        if 'predicted_activity' not in final_df.columns:
            final_df['predicted_activity'] = 0.0

        # 排序：优先使用AI预测分，否则使用生物学规则分
        sort_key = 'predicted_activity' if self.predictor else 'biological_score'
        final_df = final_df.sort_values(by=sort_key, ascending=False)
        
        # 保存最终候选列表
        final_csv_path = os.path.join(self.results_dir, f"final_predicted_candidates_{self.timestamp}.csv")
        final_df.to_csv(final_csv_path, index=False)
        
        print(f"💾 最终候选列表已保存: {final_csv_path}")

    def _generate_comprehensive_report(self):
        """生成综合报告"""
        report_path = os.path.join(self.results_dir, f"peptide_discovery_report_{self.timestamp}.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 三方法集成多肽发现报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
            f.write(f"**运行标识**: {self.timestamp}\n")
            if self.predictor:
                f.write(f"**AI筛选模型**: {os.path.basename(self.predictor.model.name)}\n")
            else:
                f.write(f"**AI筛选模型**: 未使用\n")
            f.write("\n## 方法概述\n\n")
            f.write("本次发现采用三种互补的探索策略:\n\n")
            f.write("1. **序列变异**: 基于已知高活性序列的局部探索\n")
            f.write("2. **理性设计**: 基于生物学知识的定向设计\n")
            f.write(f"3. **VAE生成**: {'AI分类器引导的' if self.predictor else '启发式规则引导的'}创新序列发现\n\n")
            
            # 写入各方法的结果
            for method, candidates in self.method_results.items():
                f.write(f"### {method.replace('_', ' ').title()}\n\n")
                f.write(f"- 生成序列数: {len(candidates)}\n")
                
                df = pd.DataFrame(candidates)
                if not df.empty:
                    avg_len = df['sequence'].apply(len).mean()
                    f.write(f"- 平均长度: {avg_len:.1f}\n")
                    f.write(f"- 独特序列数: {df['sequence'].nunique()}\n\n")
                    
                    # 排序并选择Top 5
                    sort_key = 'predicted_activity' if 'predicted_activity' in df.columns and self.predictor else 'biological_score'
                    if sort_key in df.columns:
                        top_5 = df.sort_values(by=sort_key, ascending=False).head(5)
                    else:
                        top_5 = df.head(5)

                    f.write(f"**Top 5 序列**:\n\n")
                    for i, row in top_5.iterrows():
                        score = row.get(sort_key, 'N/A')
                        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
                        f.write(f"{i+1}. `{row['sequence']}` (评分: {score_str})\n")
                f.write("\n")
            
            f.write("## 实验建议\n\n")
            f.write("建议优先合成和测试AI预测活性概率高或生物学评分≥70的候选序列，这些序列在理论上具有良好的抗菌潜力。\n\n")
            f.write("详细结果请查看各方法的子目录和CSV文件。\n")
            
        print(f"✅ 综合报告已生成: {report_path}")

    def _remove_duplicates(self, candidates):
        """去除重复序列"""
        seen_sequences = set()
        unique_candidates = []
        
        for candidate in candidates:
            sequence = candidate['sequence']
            if sequence not in seen_sequences:
                seen_sequences.add(sequence)
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _save_method_results(self, method_name, candidates, exploration_log):
        """保存单个方法的结果"""
        method_dir = os.path.join(self.results_dir, method_name)
        
        # 保存候选序列
        df = pd.DataFrame(candidates)
        csv_path = os.path.join(method_dir, f"{method_name}_candidates_{self.timestamp}.csv")
        df.to_csv(csv_path, index=False)
        
        # 保存探索日志
        log_path = os.path.join(method_dir, f"{method_name}_log_{self.timestamp}.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(exploration_log, f, indent=2, ensure_ascii=False)
        
        print(f"✅ {method_name} 结果已保存到: {method_dir}")

def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(description='三方法集成多肽发现框架')
    parser.add_argument('--sv_count', type=int, default=150, help='序列变异生成数量')
    parser.add_argument('--rd_count', type=int, default=80, help='理性设计生成数量')
    parser.add_argument('--vae_count', type=int, default=120, help='VAE生成数量')
    parser.add_argument('--model_path', type=str, 
                        default='results/physchem_seqeng_biobert_dl_rawseq/best_physchem_seqeng_biobert_rawseq_classification.h5', 
                        help='AI活性预测模型路径 (.h5文件)')
    parser.add_argument('--scaler_path', type=str, default=None, 
                        help='AI模型标准化器路径 (.pkl文件)，可选，会自动查找')
    
    args = parser.parse_args()
    
    # 初始化发现系统，并传入AI模型路径
    discovery_system = ThreeMethodPeptideDiscovery(
        model_path=args.model_path,
        scaler_path=args.scaler_path
    )
    
    # 运行完整流程
    discovery_system.run_discovery_pipeline(
        sv_count=args.sv_count,
        rd_count=args.rd_count,
        vae_count=args.vae_count
    )

if __name__ == "__main__":
    main()
