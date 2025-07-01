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
    print("警告: VAE模块导入失败")
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
            print("=" * 60)
            print("正在加载多模态深度学习CNN分类器...")
            self.predictor = PhysChemSeqEngBioBERTPredictor(model_path, scaler_path)
            print("✅ 多模态深度学习CNN分类器加载成功！")
            print("=" * 60)
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
    
    def run_sequence_variation(self, target_count=200, mutation_intensity='medium'):
        """方法1: 序列变异 - 传统空间探索
        
        Args:
            target_count: 目标候选数量
            mutation_intensity: 变异强度 ('low', 'medium', 'high')
        """
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
                'intensity': mutation_intensity,
                'max_mutations_per_sequence': 4 if mutation_intensity == 'high' else (3 if mutation_intensity == 'medium' else 2),
                'length_range': [8, 16]
            }
        }
        
        candidates = []
        seed_sequences = high_activity['sequence'].tolist()
        
        print(f"基于 {len(seed_sequences)} 个高活性种子序列生成变异体...")
        
        for i in range(target_count):
            if seed_sequences:
                base_seq = np.random.choice(seed_sequences)
                
                # 根据变异强度调整变异次数
                if mutation_intensity == 'low':
                    num_mutations = np.random.randint(1, 3)  # 1-2次变异
                elif mutation_intensity == 'medium':
                    num_mutations = np.random.randint(1, 4)  # 1-3次变异
                else:  # high
                    num_mutations = np.random.randint(2, 5)  # 2-4次变异
                    
                mutated_seq = self._mutate_sequence(base_seq, num_mutations, mutation_intensity)
                
                # 基本质量检查
                if 8 <= len(mutated_seq) <= 16 and self._is_valid_sequence(mutated_seq):
                    candidates.append({
                        'sequence': mutated_seq,
                        'method': 'sequence_variation',
                        'base_sequence': base_seq,
                        'num_mutations': num_mutations,
                        'final_length': len(mutated_seq),
                        'exploration_strategy': 'local_search',
                        'biological_score': self._calculate_biological_score(mutated_seq)
                    })
        
        # 去重
        unique_candidates = self._remove_duplicates(candidates)
        
        # 保存结果
        self._save_method_results('sequence_variation', unique_candidates, exploration_log)
        
        print(f"✅ 序列变异完成: 生成 {len(unique_candidates)} 个独特候选序列")
        self.method_results['sequence_variation'] = unique_candidates
        
        return unique_candidates
    

    
    def run_rational_design(self, target_count=100):
        """方法2: 理性设计 - 数据驱动的结构感知设计"""
        print("=" * 60)
        print("方法2: 理性设计 - 数据驱动的结构感知设计")
        print("=" * 60)
        
        # 导入并运行数据分析
        from rational_design_peptide import main as rational_main
        
        print("使用详细理性设计分析...")
        design_results = rational_main()
        
        # 提取关键数据驱动洞察
        aa_stats = design_results['aa_stats']
        motif_activities = design_results['motif_activities']
        position_contributions = design_results['position_contributions']
        
        print("基于数据洞察进行结构感知设计...")
        candidates = []
        
        # 分析数据中的结构倾向性
        structure_insights = self._analyze_structural_preferences(aa_stats, motif_activities)
        
        # 设计分配：每种结构模式使用不同比例
        helix_count = int(target_count * 0.4)    # 40% - 主导模式
        sheet_count = int(target_count * 0.35)   # 35% - 次要模式
        coil_count = target_count - helix_count - sheet_count  # 25% - 灵活模式
        
        # 1. 数据驱动的两亲性螺旋设计
        helix_candidates = self._design_data_driven_helix(
            helix_count, aa_stats, motif_activities, position_contributions
        )
        candidates.extend(helix_candidates)
        
        # 2. 数据驱动的β折叠设计
        sheet_candidates = self._design_data_driven_sheet(
            sheet_count, aa_stats, motif_activities, structure_insights
        )
        candidates.extend(sheet_candidates)
        
        # 3. 数据驱动的无规卷曲设计
        coil_candidates = self._design_data_driven_coil(
            coil_count, aa_stats, motif_activities
        )
        candidates.extend(coil_candidates)
        
        # 去重和保存
        unique_candidates = self._remove_duplicates(candidates)
        exploration_log = {
            'method': 'rational_design',
            'strategy': 'data_driven_structure_aware_design',
            'timestamp': self.timestamp,
            'target_count': target_count,
            'actual_count': len(unique_candidates),
            'design_distribution': {
                'amphipathic_helix': len(helix_candidates),
                'beta_sheet': len(sheet_candidates),
                'random_coil': len(coil_candidates)
            },
            'structural_insights': structure_insights
        }
        
        self._save_method_results('rational_design', unique_candidates, exploration_log)
        
        print(f"✅ 理性设计完成: 生成 {len(unique_candidates)} 个结构感知序列")
        print(f"   - 两亲性螺旋: {len(helix_candidates)} 个")
        print(f"   - β折叠结构: {len(sheet_candidates)} 个") 
        print(f"   - 无规卷曲: {len(coil_candidates)} 个")
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
                'training_epochs': 30,
                'temperature_sampling': 1.0,
                'feedback_enabled': True,
                'ai_predictor_integrated': bool(self.predictor),
                'ai_score_threshold': ai_score_threshold if self.predictor else 'N/A'
            }
        }
        
        candidates = []
        
        if VAEPeptideGenerator is not None:
            print("使用VAE模型进行生成...")
            
            # 将AI预测器注入VAE生成器
            vae_generator = VAEPeptideGenerator(predictor=self.predictor)
            
            # 快速训练模式
            print("准备训练数据...")
            vae_generator.prepare_data()
            
            print("构建并训练VAE模型...")
            vae_generator.build_model(latent_dim=16)
            vae_generator.train_model(epochs=30, batch_size=8)
            
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
                    'generation_strategy': 'ai_guided_latent_space_sampling' if self.predictor else 'latent_space_sampling',
                    'biological_score': self._calculate_biological_score(item['sequence'])
                })
            
            exploration_log['vae_training_successful'] = True
            
        else:
            print("VAE模块不可用，使用高级启发式生成...")
            candidates = self._advanced_heuristic_generation(target_count)
            exploration_log['vae_training_successful'] = False
        
        # 去重和质量筛选
        unique_candidates = self._remove_duplicates(candidates)
        
        # 保存结果
        self._save_method_results('vae_generation', unique_candidates, exploration_log)
        
        print(f"✅ VAE生成完成: 生成 {len(unique_candidates)} 个创新序列")
        self.method_results['vae_generation'] = unique_candidates
        
        return unique_candidates
    
    def _predict_activity_for_all_candidates(self, candidates):
        """使用加载的AI模型为所有候选序列预测活性"""
        if not self.predictor:
            print("AI预测器不可用，跳过最终AI预测。")
            return candidates
            
        if not candidates:
            print("没有候选序列，跳过预测。")
            return candidates
        
        print("=" * 60)
        print("最终筛选: 使用多模态深度学习CNN分类器对所有候选序列进行活性预测...")
        print("=" * 60)
        
        sequences = [c['sequence'] for c in candidates]
        
        y_prob, _ = self.predictor.predict(sequences)
        
        # 更新每个候选序列的预测活性
        for i, candidate in enumerate(candidates):
            candidate['predicted_activity'] = round(float(y_prob[i]), 4)
        
        print(f"✅ 已为 {len(candidates)} 个序列更新AI预测活性。")
        
        return candidates

    def _mutate_sequence(self, sequence, num_mutations, mutation_intensity='medium'):
        """序列变异操作
        
        Args:
            sequence: 原始序列
            num_mutations: 变异次数
            mutation_intensity: 变异强度 ('low', 'medium', 'high')
        """
        seq_list = list(sequence)
        amino_acids = ['R', 'W', 'K', 'I', 'V', 'F', 'Y', 'L', 'A']
        
        # 根据变异强度调整变异类型权重
        if mutation_intensity == 'low':
            # 低强度：主要替换，少量插入删除
            mutation_weights = [0.7, 0.15, 0.15]  # [substitute, insert, delete]
        elif mutation_intensity == 'medium':
            # 中等强度：各类型平衡
            mutation_weights = [0.5, 0.25, 0.25]
        else:  # high
            # 高强度：更多插入删除
            mutation_weights = [0.4, 0.3, 0.3]
            
        mutation_types = ['substitute', 'insert', 'delete']
        
        for _ in range(num_mutations):
            mutation_type = np.random.choice(mutation_types, p=mutation_weights)
            
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
        """设计两亲性螺旋结构
        
        科学原理:
        - α螺旋的疏水面插入细菌膜脂质双分子层
        - 亲水面与膜表面磷脂头基团形成静电相互作用
        - 3.6个残基/圈的周期性，疏水面约占180度
        - 典型代表: Magainin, Cecropin, LL-37
        """
        # 严格的两亲性设计：疏水面和亲水面分离
        hydrophobic = ['I', 'L', 'V', 'F', 'W', 'Y']  # 疏水侧链
        hydrophilic = ['R', 'K', 'H', 'S', 'T', 'N', 'Q']  # 亲水侧链
        
        length = np.random.randint(10, 15)
        sequence = ""
        
        for i in range(length):
            # 基于螺旋轮的两亲性设计 (100度/残基)
            helix_angle = (i * 100) % 360
            
            if helix_angle < 180:  # 疏水面 (0-180度)
                sequence += np.random.choice(hydrophobic)
            else:  # 亲水面 (180-360度)
                sequence += np.random.choice(hydrophilic)
        
        return sequence
    
    def _design_beta_sheet(self):
        """设计β折叠结构
        
        科学原理:
        - 多个β链聚集形成β桶状跨膜孔道
        - 高疏水性的β链插入膜核心区域
        - 正电荷残基稳定负电荷的细菌膜
        - 典型代表: Protegrin, Tachyplesin, Defensin
        """
        # β折叠倾向氨基酸 (基于Ramachandran图分析)
        beta_preferred = ['I', 'V', 'F', 'Y', 'W', 'K', 'R']
        
        length = np.random.randint(10, 14)
        sequence = ""
        
        for i in range(length):
            sequence += np.random.choice(beta_preferred)
        
        # 确保足够正电荷与细菌膜相互作用 (PE/PG含量高)
        positive_count = sequence.count('R') + sequence.count('K')
        if positive_count < 2:
            seq_list = list(sequence)
            for _ in range(2 - positive_count):
                pos = np.random.randint(0, len(seq_list))
                # 优先使用精氨酸 (胍基团的多重氢键)
                seq_list[pos] = np.random.choice(['R', 'K'], p=[0.7, 0.3])
            sequence = ''.join(seq_list)
        
        return sequence
    
    def _design_random_coil(self):
        """设计无规卷曲结构
        
        科学原理:
        - 高度灵活的构象适应不同膜环境
        - 通过构象变化实现膜结合到膜穿透的转变
        - 富含脯氨酸和甘氨酸等破坏规则结构的残基
        - 典型代表: Indolicidin, 一些富含脯氨酸的抗菌肽
        """
        # 平衡组成：功能性氨基酸 + 结构破坏子
        amino_acids = ['R', 'K', 'W', 'F', 'I', 'V', 'L', 'A', 'G', 'Y']
        # 权重基于功能重要性和构象灵活性需求
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
                bio_score = self._calculate_biological_score(sequence)
                candidates.append({
                    'sequence': sequence,
                    'method': 'vae_generation',
                    'generation_strategy': strategy,
                    'predicted_activity': bio_score / 100,
                    'exploration_strategy': 'heuristic_simulation',
                    'biological_score': bio_score
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

    def run_discovery_pipeline(self, sv_count=200, rd_count=100, vae_count=150, mutation_intensity='medium'):
        """运行完整的多肽发现流程
        
        Args:
            sv_count: 序列变异生成数量
            rd_count: 理性设计生成数量
            vae_count: VAE生成数量
            mutation_intensity: 序列变异强度 ('low', 'medium', 'high')
        """
        print("🚀 开始三方法集成多肽发现流程...")
        print(f"序列变异强度设置: {mutation_intensity}")
        
        # 运行各个生成方法
        sv_candidates = self.run_sequence_variation(target_count=sv_count, mutation_intensity=mutation_intensity)
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
                f.write(f"**AI筛选模型**: 多模态深度学习CNN分类器\n")
            else:
                f.write(f"**AI筛选模型**: 未使用，采用生物学规则评分\n")
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
                    sort_key = 'predicted_activity' if self.predictor else 'biological_score'
                    top_5 = df.sort_values(by=sort_key, ascending=False).head(5)

                    f.write(f"**Top 5 序列**:\n\n")
                    for i, row in top_5.iterrows():
                        score = row.get(sort_key, 0)
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

    def _analyze_structural_preferences(self, aa_stats, motif_activities):
        """分析数据中的结构偏好性"""
        print("分析数据中的结构倾向性...")
        
        # 1. 基于氨基酸偏好分析结构倾向
        helix_formers = ['A', 'E', 'L', 'M']  # α螺旋形成氨基酸
        sheet_formers = ['V', 'I', 'F', 'Y', 'W']  # β折叠形成氨基酸
        turn_formers = ['G', 'P', 'S', 'D', 'N']  # 转角/无规形成氨基酸
        
        # 计算高活性氨基酸的结构倾向
        high_activity_aa = [aa for aa, stats in aa_stats.items() 
                           if stats['mean_activity'] > np.mean([s['mean_activity'] for s in aa_stats.values()])]
        
        helix_score = sum(aa_stats[aa]['mean_activity'] for aa in high_activity_aa if aa in helix_formers)
        sheet_score = sum(aa_stats[aa]['mean_activity'] for aa in high_activity_aa if aa in sheet_formers)
        turn_score = sum(aa_stats[aa]['mean_activity'] for aa in high_activity_aa if aa in turn_formers)
        
        # 2. 分析motif的结构特征
        helix_motifs = []
        sheet_motifs = []
        
        for motif, activities in motif_activities.items():
            if len(activities) >= 3 and np.mean(activities) > 0.5:
                # 简单的结构分类：基于疏水性模式
                hydrophobic_count = sum(1 for aa in motif if aa in 'ILMFWYV')
                charged_count = sum(1 for aa in motif if aa in 'RKDE')
                
                if len(motif) >= 3:
                    # 两亲性模式倾向于螺旋
                    if hydrophobic_count > 0 and charged_count > 0:
                        helix_motifs.append((motif, np.mean(activities)))
                    # 高疏水性倾向于折叠
                    elif hydrophobic_count >= len(motif) * 0.6:
                        sheet_motifs.append((motif, np.mean(activities)))
        
        # 排序获取最佳motif
        helix_motifs.sort(key=lambda x: x[1], reverse=True)
        sheet_motifs.sort(key=lambda x: x[1], reverse=True)
        
        insights = {
            'preferred_structure': 'helix' if helix_score > sheet_score else 'sheet',
            'helix_preference_score': helix_score,
            'sheet_preference_score': sheet_score,
            'turn_preference_score': turn_score,
            'top_helix_motifs': [motif for motif, _ in helix_motifs[:10]],
            'top_sheet_motifs': [motif for motif, _ in sheet_motifs[:10]],
            'high_activity_aa': high_activity_aa
        }
        
        print(f"结构偏好分析: {insights['preferred_structure']} (螺旋:{helix_score:.1f}, 折叠:{sheet_score:.1f})")
        
        return insights

    def _design_data_driven_helix(self, count, aa_stats, motif_activities, position_contributions):
        """数据驱动的两亲性螺旋设计"""
        print(f"设计 {count} 个数据驱动的两亲性螺旋...")
        
        candidates = []
        
        # 获取高活性的疏水性和亲水性氨基酸
        high_activity_aa = [aa for aa, stats in aa_stats.items() 
                           if stats['mean_activity'] > np.mean([s['mean_activity'] for s in aa_stats.values()])]
        
        hydrophobic_aa = [aa for aa in high_activity_aa if aa in 'ILMFWYV']
        hydrophilic_aa = [aa for aa in high_activity_aa if aa in 'RKHDNQST']
        
        # 从数据中提取两亲性motif
        amphipathic_motifs = []
        for motif, activities in motif_activities.items():
            if len(activities) >= 3 and np.mean(activities) > 0.5 and len(motif) >= 3:
                hydrophobic_count = sum(1 for aa in motif if aa in 'ILMFWYV')
                hydrophilic_count = sum(1 for aa in motif if aa in 'RKHDNQST')
                if hydrophobic_count > 0 and hydrophilic_count > 0:
                    amphipathic_motifs.append(motif)
        
        for i in range(count):
            # 螺旋周期性设计 (3.6个残基/圈)
            length = np.random.randint(10, 15)
            sequence = ""
            
            # 30%概率使用数据驱动的motif
            if amphipathic_motifs and np.random.random() < 0.3:
                motif = np.random.choice(amphipathic_motifs)
                sequence += motif
                remaining = length - len(motif)
            else:
                remaining = length
            
            # 补充序列：考虑螺旋面的两亲性
            for pos in range(remaining):
                # 螺旋面设计：100度相位差
                helix_position = (len(sequence) * 100) % 360
                
                if helix_position < 180:  # 疏水面
                    if hydrophobic_aa:
                        weights = [aa_stats[aa]['mean_activity'] for aa in hydrophobic_aa]
                        sequence += np.random.choice(hydrophobic_aa, p=np.array(weights)/sum(weights))
                    else:
                        sequence += np.random.choice(['I', 'L', 'V', 'F'])
                else:  # 亲水面
                    if hydrophilic_aa:
                        weights = [aa_stats[aa]['mean_activity'] for aa in hydrophilic_aa]
                        sequence += np.random.choice(hydrophilic_aa, p=np.array(weights)/sum(weights))
                    else:
                        sequence += np.random.choice(['R', 'K', 'H'])
            
            if self._is_valid_sequence(sequence):
                candidates.append({
                    'sequence': sequence,
                    'method': 'rational_design_enhanced',
                    'design_pattern': 'data_driven_amphipathic_helix',
                    'biological_score': self._calculate_biological_score(sequence),
                    'structure_rationale': 'helix_amphipathic_periodicity',
                    'data_integration': 'motif_and_aa_preferences'
                })
        
        return candidates
    
    def _design_data_driven_sheet(self, count, aa_stats, motif_activities, structure_insights):
        """数据驱动的β折叠设计"""
        print(f"设计 {count} 个数据驱动的β折叠...")
        
        candidates = []
        
        # 获取高活性的β折叠倾向氨基酸
        sheet_preferred = [aa for aa in structure_insights['high_activity_aa'] if aa in 'IVFYWR']
        if not sheet_preferred:
            sheet_preferred = ['I', 'V', 'F', 'Y', 'W', 'R']
        
        # 获取β折叠相关的高活性motif
        sheet_motifs = structure_insights['top_sheet_motifs']
        
        for i in range(count):
            length = np.random.randint(10, 14)
            sequence = ""
            
            # 40%概率使用数据中的β折叠motif
            if sheet_motifs and np.random.random() < 0.4:
                motif = np.random.choice(sheet_motifs)
                sequence += motif
                remaining = length - len(motif)
            else:
                remaining = length
            
            # 补充序列：β折叠设计原则
            for pos in range(remaining):
                # β折叠交替模式：疏水-亲水-疏水
                if pos % 2 == 0:  # 疏水位置
                    hydrophobic_sheet = [aa for aa in sheet_preferred if aa in 'IVFYW']
                    if hydrophobic_sheet:
                        weights = [aa_stats[aa]['mean_activity'] for aa in hydrophobic_sheet]
                        sequence += np.random.choice(hydrophobic_sheet, p=np.array(weights)/sum(weights))
                    else:
                        sequence += np.random.choice(['I', 'V', 'F'])
                else:  # 极性位置
                    polar_sheet = [aa for aa in sheet_preferred if aa in 'RK']
                    if polar_sheet:
                        weights = [aa_stats[aa]['mean_activity'] for aa in polar_sheet]
                        sequence += np.random.choice(polar_sheet, p=np.array(weights)/sum(weights))
                    else:
                        sequence += np.random.choice(['R', 'K'])
            
            # 确保正电荷（基于数据驱动的需求）
            positive_count = sequence.count('R') + sequence.count('K')
            if positive_count < 2:
                # 基于数据选择最佳正电荷氨基酸
                best_positive = 'R' if aa_stats.get('R', {}).get('mean_activity', 0) > aa_stats.get('K', {}).get('mean_activity', 0) else 'K'
                seq_list = list(sequence)
                for _ in range(2 - positive_count):
                    pos = np.random.randint(0, len(seq_list))
                    seq_list[pos] = best_positive
                sequence = ''.join(seq_list)
            
            if self._is_valid_sequence(sequence):
                candidates.append({
                    'sequence': sequence,
                    'method': 'rational_design_enhanced',
                    'design_pattern': 'data_driven_beta_sheet',
                    'biological_score': self._calculate_biological_score(sequence),
                    'structure_rationale': 'sheet_strand_alignment',
                    'data_integration': 'beta_motifs_and_preferences'
                })
        
        return candidates
    
    def _design_data_driven_coil(self, count, aa_stats, motif_activities):
        """数据驱动的无规卷曲设计"""
        print(f"设计 {count} 个数据驱动的无规卷曲...")
        
        candidates = []
        
        # 基于数据的最优氨基酸分布
        high_activity_aa = [aa for aa, stats in aa_stats.items() 
                           if stats['mean_activity'] > np.mean([s['mean_activity'] for s in aa_stats.values()])]
        
        # 计算数据驱动的权重分布
        aa_weights = {}
        total_activity = sum(aa_stats[aa]['mean_activity'] for aa in high_activity_aa)
        for aa in high_activity_aa:
            aa_weights[aa] = aa_stats[aa]['mean_activity'] / total_activity
        
        # 获取高活性的灵活性motif（短motif）
        flexible_motifs = [motif for motif, activities in motif_activities.items() 
                          if len(activities) >= 3 and np.mean(activities) > 0.5 and len(motif) <= 3]
        
        for i in range(count):
            length = np.random.randint(10, 14)
            sequence = ""
            
            # 25%概率使用数据中的短motif
            if flexible_motifs and np.random.random() < 0.25:
                motif = np.random.choice(flexible_motifs)
                sequence += motif
                remaining = length - len(motif)
            else:
                remaining = length
            
            # 按数据驱动的分布填充
            for pos in range(remaining):
                # 添加一些位置特异性考虑
                if pos < 3 or pos >= length - 3:  # 末端位置
                    # 末端倾向于使用高活性的极性氨基酸
                    terminal_aa = [aa for aa in high_activity_aa if aa in 'RKHNQST']
                    if terminal_aa:
                        weights = [aa_weights[aa] for aa in terminal_aa]
                        sequence += np.random.choice(terminal_aa, p=np.array(weights)/sum(weights))
                    else:
                        sequence += np.random.choice(high_activity_aa, p=list(aa_weights.values()))
                else:  # 中间位置
                    sequence += np.random.choice(high_activity_aa, p=list(aa_weights.values()))
            
            if self._is_valid_sequence(sequence):
                candidates.append({
                    'sequence': sequence,
                    'method': 'rational_design_enhanced',
                    'design_pattern': 'data_driven_random_coil',
                    'biological_score': self._calculate_biological_score(sequence),
                    'structure_rationale': 'flexible_functional_segments',
                    'data_integration': 'activity_weighted_composition'
                })
        
        return candidates
def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(description='三方法集成多肽发现框架')
    parser.add_argument('--sv_count', type=int, default=150, help='序列变异生成数量')
    parser.add_argument('--rd_count', type=int, default=80, help='理性设计生成数量')
    parser.add_argument('--vae_count', type=int, default=120, help='VAE生成数量')
    parser.add_argument('--mutation_intensity', type=str, default='medium', 
                        choices=['low', 'medium', 'high'],
                        help='序列变异强度 (low: 保守, medium: 中等, high: 激进)')
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
        vae_count=args.vae_count,
        mutation_intensity=args.mutation_intensity
    )

if __name__ == "__main__":
    main()
