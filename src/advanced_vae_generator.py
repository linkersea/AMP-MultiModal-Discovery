#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真正的VAE（变分自编码器）多肽生成模型
结合分类器反馈的智能序列生成系统
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import json
import os
from datetime import datetime
from collections import Counter

class PeptideDataset(Dataset):
    """多肽序列数据集"""
    def __init__(self, sequences, activities=None, max_length=20):
        self.sequences = sequences
        self.activities = activities
        self.max_length = max_length
        
        # 创建氨基酸到索引的映射
        unique_aas = set(''.join(sequences))
        self.aa_to_idx = {aa: i+1 for i, aa in enumerate(sorted(unique_aas))}
        self.aa_to_idx['<PAD>'] = 0  # 填充符
        self.aa_to_idx['<START>'] = len(self.aa_to_idx)
        self.aa_to_idx['<END>'] = len(self.aa_to_idx)
        
        self.idx_to_aa = {v: k for k, v in self.aa_to_idx.items()}
        self.vocab_size = len(self.aa_to_idx)
        
        print(f"词汇表大小: {self.vocab_size}")
        print(f"氨基酸映射: {self.aa_to_idx}")
    
    def encode_sequence(self, sequence):
        """将序列编码为索引"""
        encoded = [self.aa_to_idx['<START>']]
        for aa in sequence:
            if aa in self.aa_to_idx:
                encoded.append(self.aa_to_idx[aa])
        encoded.append(self.aa_to_idx['<END>'])
        
        # 填充到固定长度
        if len(encoded) < self.max_length:
            encoded.extend([self.aa_to_idx['<PAD>']] * (self.max_length - len(encoded)))
        else:
            encoded = encoded[:self.max_length]
        
        return encoded
    
    def decode_sequence(self, encoded):
        """将索引解码为序列"""
        sequence = ""
        for idx in encoded:
            if idx == self.aa_to_idx['<PAD>'] or idx == self.aa_to_idx['<END>']:
                break
            if idx != self.aa_to_idx['<START>'] and idx in self.idx_to_aa:
                sequence += self.idx_to_aa[idx]
        return sequence
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        encoded_seq = torch.tensor(self.encode_sequence(self.sequences[idx]), dtype=torch.long)
        
        if self.activities is not None:
            activity = torch.tensor(self.activities[idx], dtype=torch.float)
            return encoded_seq, activity
        
        return encoded_seq

class PeptideVAE(nn.Module):
    """多肽VAE模型"""
    def __init__(self, vocab_size, max_length, embedding_dim=64, hidden_dim=128, latent_dim=32):
        super(PeptideVAE, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 编码器
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
        # 解码器
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    def encode(self, x):
        """编码器：序列 -> 潜在变量分布参数"""
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM编码
        lstm_out, (hidden, _) = self.encoder_lstm(embedded)
        
        # 使用最后一个时间步的隐藏状态
        # hidden shape: (2, batch_size, hidden_dim) -> (batch_size, hidden_dim*2)
        hidden = hidden.transpose(0, 1).contiguous().view(-1, self.hidden_dim * 2)
        
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, max_length=None):
        """解码器：潜在变量 -> 序列"""
        if max_length is None:
            max_length = self.max_length
        
        batch_size = z.size(0)
        
        # 初始化解码器输入
        decoder_input = self.decoder_input(z).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # 初始化LSTM状态
        h_0 = torch.zeros(1, batch_size, self.hidden_dim).to(z.device)
        c_0 = torch.zeros(1, batch_size, self.hidden_dim).to(z.device)
        
        outputs = []
        hidden = (h_0, c_0)
        
        for t in range(max_length):
            lstm_out, hidden = self.decoder_lstm(decoder_input, hidden)
            output = self.output_projection(lstm_out.squeeze(1))
            outputs.append(output)
            
            # 使用上一步的输出作为下一步的输入（训练时使用teacher forcing）
            decoder_input = lstm_out
        
        # outputs: list of (batch_size, vocab_size) -> (batch_size, max_length, vocab_size)
        return torch.stack(outputs, dim=1)
    
    def forward(self, x):
        """前向传播"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """VAE损失函数"""
    # 重构损失（交叉熵）
    recon_loss = F.cross_entropy(recon_x.view(-1, recon_x.size(-1)), x.view(-1), ignore_index=0)
    
    # KL散度损失
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss /= x.size(0) * x.size(1)  # 标准化
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

class VAEPeptideGenerator:
    """VAE多肽生成器"""
    def __init__(self, model_save_path='vae_peptide_model.pth', predictor=None):
        self.model = None
        self.dataset = None
        self.model_save_path = model_save_path
        self.training_history = []
        self.predictor = predictor  # 接收外部预测器
        
        if self.predictor:
            print("✅ VAE生成器已与外部AI分类器集成！")

    def _get_feedback_score(self, sequence):
        """获取序列的反馈评分"""
        # 如果集成了外部预测器，则使用它
        if self.predictor:
            try:
                y_prob, _ = self.predictor.predict([sequence])
                return y_prob[0]
            except Exception as e:
                print(f"⚠️ 预测器评分失败: {e}，回退到基础评分。")
                # Fallback to basic scoring on error
        
        # 默认的模拟实现: 使用一个简化的生物学评分
        charge = sequence.count('K') + sequence.count('R')
        hydrophobic_ratio = sum(1 for aa in sequence if aa in 'AILMFWYV') / len(sequence)
        
        score = 0.0
        if 2 <= charge <= 8: score += 0.5
        if 0.3 <= hydrophobic_ratio <= 0.6: score += 0.5
            
        return score

    def generate_peptides_with_feedback(self, num_samples, temperature=1.0, 
                                      max_attempts=5, score_threshold=0.5):
        """使用外部反馈评分生成高质量多肽"""
        if not self.model:
            raise Exception("模型未训练或加载！")
        
        print(f"生成 {num_samples} 个多肽序列 (评分阈值 > {score_threshold})...")
        
        high_quality_peptides = []
        generated_sequences = set()
        
        for _ in range(num_samples * max_attempts):
            # 1. VAE生成一个候选序列
            generated_sequence = self._generate_single_sequence(temperature=temperature)
            
            if generated_sequence and generated_sequence not in generated_sequences:
                generated_sequences.add(generated_sequence)
                
                # 2. 获取外部反馈评分
                feedback_score = self._get_feedback_score(generated_sequence)
                
                # 3. 根据评分决定是否接受
                if feedback_score >= score_threshold:
                    high_quality_peptides.append({
                        'sequence': generated_sequence,
                        'method': 'vae_generation',
                        'predicted_activity': feedback_score,
                        'generation_temperature': temperature,
                        'exploration_strategy': 'ai_guided_latent_space_sampling'
                    })
                    
                    if len(high_quality_peptides) >= num_samples:
                        break
        
        print(f"生成了 {len(high_quality_peptides)} 个候选序列")
        return high_quality_peptides

    def prepare_data(self, csv_file='data/raw/120dataset.csv'):
        """准备训练数据"""
        print("准备训练数据...")
        
        df = pd.read_csv(csv_file)
        sequences = df['sequence'].tolist()
        activities = df['activity'].tolist() if 'activity' in df.columns else None
        
        # 过滤过长或过短的序列
        filtered_sequences = []
        filtered_activities = []
        
        for i, seq in enumerate(sequences):
            if 6 <= len(seq) <= 20:  # 合理的多肽长度范围
                filtered_sequences.append(seq)
                if activities:
                    filtered_activities.append(activities[i])
        
        print(f"原始序列数: {len(sequences)}")
        print(f"过滤后序列数: {len(filtered_sequences)}")
        
        # 创建数据集
        self.dataset = PeptideDataset(
            filtered_sequences, 
            filtered_activities if activities else None,
            max_length=22  # 包含START和END token
        )
        
        return self.dataset
    
    def build_model(self, embedding_dim=64, hidden_dim=128, latent_dim=32):
        """构建VAE模型"""
        print("构建VAE模型...")
        
        if self.dataset is None:
            raise ValueError("请先调用prepare_data()准备数据")
        
        self.model = PeptideVAE(
            vocab_size=self.dataset.vocab_size,
            max_length=self.dataset.max_length,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )
        
        print(f"模型参数: vocab_size={self.dataset.vocab_size}, latent_dim={latent_dim}")
        return self.model
    
    def train_model(self, epochs=50, batch_size=16, learning_rate=1e-3, beta=1.0):
        """训练VAE模型"""
        print(f"开始训练VAE模型... epochs={epochs}")
        
        if self.model is None:
            raise ValueError("请先调用build_model()构建模型")
        
        # 数据加载器
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        # 优化器
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 训练循环
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            
            for batch_idx, batch_data in enumerate(dataloader):
                if len(batch_data) == 2:  # 有活性数据
                    sequences, activities = batch_data
                else:
                    sequences = batch_data
                
                optimizer.zero_grad()
                
                # 前向传播
                recon_sequences, mu, logvar, z = self.model(sequences)
                
                # 计算损失
                loss, recon_loss, kl_loss = vae_loss_function(
                    recon_sequences, sequences, mu, logvar, beta
                )
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
            
            avg_loss = total_loss / len(dataloader)
            avg_recon_loss = total_recon_loss / len(dataloader)
            avg_kl_loss = total_kl_loss / len(dataloader)
            
            self.training_history.append({
                'epoch': epoch,
                'total_loss': avg_loss,
                'recon_loss': avg_recon_loss,
                'kl_loss': avg_kl_loss
            })
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f} (Recon={avg_recon_loss:.4f}, KL={avg_kl_loss:.4f})")
        
        print("训练完成!")
        return self.training_history
    
    def save_model(self):
        """保存模型"""
        if self.model is None:
            raise ValueError("没有训练好的模型可以保存")
        
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.dataset.vocab_size,
            'max_length': self.dataset.max_length,
            'aa_to_idx': self.dataset.aa_to_idx,
            'idx_to_aa': self.dataset.idx_to_aa,
            'training_history': self.training_history
        }
        
        torch.save(model_state, self.model_save_path)
        print(f"模型已保存到: {self.model_save_path}")
    
    def load_model(self, model_path=None):
        """加载模型"""
        if model_path is None:
            model_path = self.model_save_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model_state = torch.load(model_path, map_location='cpu')
        
        # 重建数据集信息
        class DummyDataset:
            def __init__(self, vocab_size, max_length, aa_to_idx, idx_to_aa):
                self.vocab_size = vocab_size
                self.max_length = max_length
                self.aa_to_idx = aa_to_idx
                self.idx_to_aa = idx_to_aa
        
        self.dataset = DummyDataset(
            model_state['vocab_size'],
            model_state['max_length'],
            model_state['aa_to_idx'],
            model_state['idx_to_aa']
        )
        
        # 重建模型
        self.model = PeptideVAE(
            vocab_size=self.dataset.vocab_size,
            max_length=self.dataset.max_length
        )
        self.model.load_state_dict(model_state['model_state_dict'])
        self.training_history = model_state.get('training_history', [])
        
        print(f"模型已从 {model_path} 加载")
    
    def _sample_sequence_from_logits(self, logits, temperature=1.0):
        """从logits采样序列"""
        sequence = ""
        
        for t in range(logits.size(0)):
            # 应用温度缩放
            scaled_logits = logits[t] / temperature
            probs = F.softmax(scaled_logits, dim=0)
            
            # 采样下一个token
            next_token = torch.multinomial(probs, 1).item()
            
            # 检查结束符或填充符
            if next_token == self.dataset.aa_to_idx.get('<END>', -1):
                break
            if next_token == self.dataset.aa_to_idx.get('<PAD>', -1):
                continue
            if next_token == self.dataset.aa_to_idx.get('<START>', -1):
                continue
            
            # 添加氨基酸到序列
            if next_token in self.dataset.idx_to_aa:
                sequence += self.dataset.idx_to_aa[next_token]
        
        return sequence
    
    def _apply_feedback_filtering(self, candidates, feedback_model, target_activity, num_samples):
        """应用分类器反馈进行筛选"""
        print("应用分类器反馈筛选...")
        
        # 这里可以集成实际的分类器模型
        # 暂时使用启发式评分作为代替
        scored_candidates = []
        
        for candidate in candidates:
            seq = candidate['sequence']
            
            # 简单的启发式评分（实际应用中替换为真实的分类器预测）
            score = self._heuristic_activity_score(seq)
            
            candidate['predicted_activity'] = score
            candidate['feedback_score'] = score
            scored_candidates.append(candidate)
        
        # 按评分排序并选择top candidates
        scored_candidates.sort(key=lambda x: x['feedback_score'], reverse=True)
        
        return scored_candidates[:num_samples]
    
    def _heuristic_activity_score(self, sequence):
        """启发式活性评分（临时替代分类器）"""
        score = 0
        length = len(sequence)
        
        # 正电荷氨基酸
        positive_charge = sequence.count('R') + sequence.count('K') + sequence.count('H')
        score += positive_charge * 0.15
        
        # 疏水性氨基酸
        hydrophobic = sum(sequence.count(aa) for aa in 'ILMFWYV')
        score += (hydrophobic / length) * 0.3
        
        # 芳香族氨基酸
        aromatic = sequence.count('F') + sequence.count('W') + sequence.count('Y')
        score += (aromatic / length) * 0.2
        
        # 长度偏好
        if 10 <= length <= 14:
            score += 0.2
        
        return min(score, 1.0)
    
    def interpolate_sequences(self, seq1, seq2, num_steps=5):
        """在两个序列间进行潜在空间插值"""
        if self.model is None:
            raise ValueError("请先训练或加载模型")
        
        self.model.eval()
        
        # 编码两个序列到潜在空间
        encoded1 = torch.tensor(self.dataset.encode_sequence(seq1)).unsqueeze(0)
        encoded2 = torch.tensor(self.dataset.encode_sequence(seq2)).unsqueeze(0)
        
        with torch.no_grad():
            mu1, _ = self.model.encode(encoded1)
            mu2, _ = self.model.encode(encoded2)
            
            # 在潜在空间中插值
            interpolated_sequences = []
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                
                # 解码插值后的潜在变量
                output_logits = self.model.decode(z_interp)
                sequence = self._sample_sequence_from_logits(output_logits[0])
                
                interpolated_sequences.append({
                    'sequence': sequence,
                    'alpha': alpha,
                    'method': 'latent_interpolation'
                })
        
        return interpolated_sequences

    def _generate_single_sequence(self, temperature=1.0):
        """从潜在空间生成单个序列"""
        if not self.model:
            return None
            
        self.model.eval()
        with torch.no_grad():
            # 从潜在空间采样
            z = torch.randn(1, self.model.latent_dim) * temperature
            
            # 解码生成序列
            output_logits = self.model.decode(z)
            
            # 从logits采样序列
            sequence = self._sample_sequence_from_logits(output_logits[0], temperature)
            
            return sequence if sequence and 6 <= len(sequence) <= 20 else None

def main():
    """演示VAE多肽生成"""
    print("=" * 60)
    print("VAE多肽生成模型演示")
    print("=" * 60)
    
    # 创建生成器
    generator = VAEPeptideGenerator()
    
    try:
        # 准备数据
        dataset = generator.prepare_data()
        
        # 构建模型
        model = generator.build_model(latent_dim=16)  # 较小的潜在维度用于快速训练
        
        # 训练模型
        history = generator.train_model(epochs=50, batch_size=8)
        
        # 保存模型
        generator.save_model()
        
        # 生成新序列
        generated = generator.generate_peptides_with_feedback(num_samples=50)
        
        # 保存结果
        results_df = pd.DataFrame(generated)
        output_file = 'vae_generated_peptides_enhanced.csv'
        results_df.to_csv(output_file, index=False)
        
        print(f"\n生成完成!")
        print(f"生成序列数: {len(generated)}")
        print(f"结果已保存到: {output_file}")
        
        # 显示一些生成的序列
        print(f"\n示例生成序列:")
        for i, item in enumerate(generated[:10]):
            print(f"{i+1:2d}. {item['sequence']} (评分: {item.get('feedback_score', 0):.3f})")
        
        # 演示序列插值
        if len(dataset.sequences) >= 2:
            print(f"\n演示序列插值:")
            seq1, seq2 = dataset.sequences[0], dataset.sequences[1]
            interpolated = generator.interpolate_sequences(seq1, seq2, num_steps=3)
            
            print(f"起始序列: {seq1}")
            for item in interpolated:
                print(f"α={item['alpha']:.1f}: {item['sequence']}")
            print(f"结束序列: {seq2}")
        
        return generator
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    generator = main()
