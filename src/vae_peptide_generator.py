#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于VAE的抗菌多肽生成模型
结合已有的分类模型进行序列优化
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class PeptideVAE:
    """抗菌多肽VAE生成模型"""
    
    def __init__(self, max_length=20, vocab_size=21, latent_dim=32):
        self.max_length = max_length
        self.vocab_size = vocab_size  # 20个氨基酸 + 1个padding
        self.latent_dim = latent_dim
        
        # 氨基酸词典
        self.aa_list = list('ACDEFGHIKLMNPQRSTVWY')
        self.aa_to_idx = {aa: i+1 for i, aa in enumerate(self.aa_list)}
        self.aa_to_idx['<PAD>'] = 0
        self.idx_to_aa = {v: k for k, v in self.aa_to_idx.items()}
        
        # VAE组件
        self.encoder = None
        self.decoder = None
        self.vae = None
        
    def encode_sequences(self, sequences):
        """将序列编码为数值矩阵"""
        encoded = []
        for seq in sequences:
            seq_encoded = []
            for aa in seq[:self.max_length]:
                seq_encoded.append(self.aa_to_idx.get(aa, 0))
            
            # Padding
            while len(seq_encoded) < self.max_length:
                seq_encoded.append(0)
            
            encoded.append(seq_encoded)
        
        return np.array(encoded)
    
    def decode_sequences(self, encoded_seqs):
        """将数值矩阵解码为序列"""
        sequences = []
        for seq_encoded in encoded_seqs:
            seq = ""
            for idx in seq_encoded:
                if idx > 0:  # 忽略padding
                    seq += self.idx_to_aa.get(idx, 'X')
            sequences.append(seq)
        return sequences
    
    def build_encoder(self):
        """构建编码器"""
        inputs = keras.Input(shape=(self.max_length,))
        
        # Embedding层
        x = layers.Embedding(self.vocab_size, 64, mask_zero=True)(inputs)
        
        # LSTM层
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.LSTM(64)(x)
        x = layers.Dropout(0.3)(x)
        
        # 全连接层
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # 均值和方差
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        
        encoder = keras.Model(inputs, [z_mean, z_log_var], name='encoder')
        return encoder
    
    def build_decoder(self):
        """构建解码器"""
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        
        # 全连接层
        x = layers.Dense(64, activation='relu')(latent_inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.RepeatVector(self.max_length)(x)
        
        # LSTM层
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.LSTM(128, return_sequences=True)(x)
        
        # 输出层
        outputs = layers.TimeDistributed(
            layers.Dense(self.vocab_size, activation='softmax')
        )(x)
        
        decoder = keras.Model(latent_inputs, outputs, name='decoder')
        return decoder
    
    def sampling(self, args):
        """重参数化技巧"""
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def build_vae(self):
        """构建完整的VAE模型"""
        # 编码器
        self.encoder = self.build_encoder()
        
        # 解码器
        self.decoder = self.build_decoder()
        
        # VAE
        inputs = keras.Input(shape=(self.max_length,))
        z_mean, z_log_var = self.encoder(inputs)
        z = layers.Lambda(self.sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])
        outputs = self.decoder(z)
        
        self.vae = keras.Model(inputs, outputs, name='vae')
        
        # 损失函数
        def vae_loss(y_true, y_pred):
            # 重构损失
            reconstruction_loss = tf.reduce_mean(
                keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            ) * self.max_length
            
            # KL散度损失
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss) * -0.5
            
            total_loss = reconstruction_loss + kl_loss
            return total_loss
        
        self.vae.compile(optimizer='adam', loss=vae_loss)
        
        return self.vae
    
    def train(self, sequences, epochs=100, batch_size=32, validation_split=0.2):
        """训练VAE模型"""
        print("准备训练数据...")
        
        # 编码序列
        X = self.encode_sequences(sequences)
        print(f"训练数据形状: {X.shape}")
        
        # 构建模型
        if self.vae is None:
            print("构建VAE模型...")
            self.build_vae()
            
        # 打印模型结构
        print("\n编码器结构:")
        self.encoder.summary()
        print("\n解码器结构:")
        self.decoder.summary()
        
        # 训练
        print(f"\n开始训练VAE模型...")
        history = self.vae.fit(
            X, X,  # 自编码器的输入和输出相同
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        return history
    
    def generate_sequences(self, n_samples=100, temperature=1.0):
        """从潜在空间采样生成新序列"""
        print(f"生成 {n_samples} 个新序列...")
        
        # 从先验分布采样
        z_samples = np.random.normal(0, temperature, (n_samples, self.latent_dim))
        
        # 解码
        generated = self.decoder.predict(z_samples)
        
        # 转换为序列
        sequences = []
        for i in range(n_samples):
            seq_probs = generated[i]
            seq_indices = np.argmax(seq_probs, axis=1)
            sequences.append(seq_indices)
        
        # 解码为氨基酸序列
        peptide_sequences = self.decode_sequences(sequences)
        
        # 过滤和清理
        clean_sequences = []
        for seq in peptide_sequences:
            # 移除padding和无效字符
            clean_seq = seq.replace('<PAD>', '').replace('X', '')
            if len(clean_seq) >= 6 and len(clean_seq) <= 20:  # 长度过滤
                clean_sequences.append(clean_seq)
        
        print(f"生成有效序列: {len(clean_sequences)}")
        return clean_sequences
    
    def interpolate_sequences(self, seq1, seq2, n_steps=10):
        """在两个序列之间进行插值"""
        # 编码序列到潜在空间
        encoded1 = self.encode_sequences([seq1])
        encoded2 = self.encode_sequences([seq2])
        
        z1_mean, _ = self.encoder.predict(encoded1)
        z2_mean, _ = self.encoder.predict(encoded2)
        
        # 插值
        interpolated_sequences = []
        for i in range(n_steps):
            alpha = i / (n_steps - 1)
            z_interp = (1 - alpha) * z1_mean + alpha * z2_mean
            
            # 解码
            generated = self.decoder.predict(z_interp)
            seq_indices = np.argmax(generated[0], axis=1)
            seq = self.decode_sequences([seq_indices])[0]
            interpolated_sequences.append(seq.replace('<PAD>', '').replace('X', ''))
        
        return interpolated_sequences
    
    def optimize_in_latent_space(self, predictor, n_iterations=1000, n_candidates=100):
        """在潜在空间中优化序列"""
        print("在潜在空间中进行优化...")
        
        best_sequences = []
        best_scores = []
        
        for iteration in range(n_iterations):
            # 采样候选潜在向量
            z_candidates = np.random.normal(0, 1, (n_candidates, self.latent_dim))
            
            # 解码为序列
            generated = self.decoder.predict(z_candidates, verbose=0)
            sequences = []
            
            for i in range(n_candidates):
                seq_indices = np.argmax(generated[i], axis=1)
                seq = self.decode_sequences([seq_indices])[0]
                clean_seq = seq.replace('<PAD>', '').replace('X', '')
                if len(clean_seq) >= 6:
                    sequences.append(clean_seq)
            
            if not sequences:
                continue
            
            # 使用分类模型预测活性
            try:
                probs, _ = predictor.predict(sequences)
                
                # 找到最佳序列
                best_idx = np.argmax(probs)
                best_seq = sequences[best_idx]
                best_prob = probs[best_idx]
                
                best_sequences.append(best_seq)
                best_scores.append(best_prob)
                
                if iteration % 100 == 0:
                    print(f"迭代 {iteration}: 最佳概率 = {best_prob:.3f}, 序列 = {best_seq}")
                    
            except Exception as e:
                continue
        
        # 返回最佳结果
        if best_scores:
            sorted_results = sorted(zip(best_sequences, best_scores), 
                                  key=lambda x: x[1], reverse=True)
            return [seq for seq, _ in sorted_results[:50]], [score for _, score in sorted_results[:50]]
        else:
            return [], []
    
    def save_model(self, filepath):
        """保存模型"""
        self.vae.save(f"{filepath}_vae.h5")
        self.encoder.save(f"{filepath}_encoder.h5")
        self.decoder.save(f"{filepath}_decoder.h5")
        print(f"模型已保存: {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        try:
            self.vae = keras.models.load_model(f"{filepath}_vae.h5", compile=False)
            self.encoder = keras.models.load_model(f"{filepath}_encoder.h5")
            self.decoder = keras.models.load_model(f"{filepath}_decoder.h5")
            print(f"模型已加载: {filepath}")
        except Exception as e:
            print(f"模型加载失败: {e}")

def train_vae_on_peptide_data():
    """在多肽数据上训练VAE"""
    print("=" * 60)
    print("训练VAE抗菌多肽生成模型")
    print("=" * 60)
    
    # 加载数据
    df = pd.read_csv('data/raw/120dataset.csv')
    sequences = df['sequence'].tolist()
    
    print(f"加载 {len(sequences)} 个训练序列")
    
    # 序列长度分析
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    print(f"序列长度范围: {min(lengths)} - {max_len}")
    
    # 初始化VAE
    vae_model = PeptideVAE(max_length=max_len, latent_dim=32)
    
    # 训练模型
    history = vae_model.train(sequences, epochs=150, batch_size=16)
    
    # 保存模型
    vae_model.save_model('results/peptide_vae_model')
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    
    plt.tight_layout()
    plt.savefig('vae_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return vae_model

def generate_vae_candidates(vae_model, predictor=None):
    """使用VAE生成候选序列"""
    print("\n" + "=" * 60)
    print("使用VAE生成候选抗菌多肽序列")
    print("=" * 60)
    
    all_candidates = []
    
    # 1. 随机生成
    print("1. 随机生成序列...")
    random_seqs = vae_model.generate_sequences(n_samples=500, temperature=1.0)
    all_candidates.extend([(seq, 'vae_random') for seq in random_seqs])
    print(f"随机生成: {len(random_seqs)} 个序列")
    
    # 2. 低温度生成（更保守）
    print("2. 保守生成序列...")
    conservative_seqs = vae_model.generate_sequences(n_samples=300, temperature=0.8)
    all_candidates.extend([(seq, 'vae_conservative') for seq in conservative_seqs])
    print(f"保守生成: {len(conservative_seqs)} 个序列")
    
    # 3. 高温度生成（更多样）
    print("3. 多样性生成序列...")
    diverse_seqs = vae_model.generate_sequences(n_samples=300, temperature=1.2)
    all_candidates.extend([(seq, 'vae_diverse') for seq in diverse_seqs])
    print(f"多样性生成: {len(diverse_seqs)} 个序列")
    
    # 4. 基于高活性序列的插值
    print("4. 高活性序列插值...")
    df = pd.read_csv('data/raw/120dataset.csv')
    high_activity_seqs = df[df['activity'] >= df['activity'].quantile(0.9)]['sequence'].tolist()
    
    interpolated_seqs = []
    for i in range(min(10, len(high_activity_seqs))):
        for j in range(i+1, min(10, len(high_activity_seqs))):
            interp_seqs = vae_model.interpolate_sequences(
                high_activity_seqs[i], high_activity_seqs[j], n_steps=5
            )
            interpolated_seqs.extend(interp_seqs)
    
    all_candidates.extend([(seq, 'vae_interpolation') for seq in interpolated_seqs])
    print(f"插值生成: {len(interpolated_seqs)} 个序列")
    
    # 5. 潜在空间优化（如果有预测器）
    if predictor is not None:
        print("5. 潜在空间优化...")
        optimized_seqs, scores = vae_model.optimize_in_latent_space(predictor, n_iterations=500)
        all_candidates.extend([(seq, 'vae_optimized') for seq in optimized_seqs])
        print(f"优化生成: {len(optimized_seqs)} 个序列")
    
    # 去重和过滤
    unique_candidates = []
    seen_sequences = set()
    
    for seq, method in all_candidates:
        if seq not in seen_sequences and 6 <= len(seq) <= 20:
            # 检查是否只包含标准氨基酸
            if all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in seq):
                unique_candidates.append((seq, method))
                seen_sequences.add(seq)
    
    print(f"\n总生成序列: {len(all_candidates)}")
    print(f"去重后有效序列: {len(unique_candidates)}")
    
    # 保存结果
    candidates_df = pd.DataFrame(unique_candidates, columns=['sequence', 'generation_method'])
    candidates_df.to_csv('vae_generated_candidates.csv', index=False)
    
    # 统计信息
    method_counts = candidates_df['generation_method'].value_counts()
    print(f"\n按生成方法分布:")
    for method, count in method_counts.items():
        print(f"  {method}: {count}")
    
    return candidates_df

def main():
    """主函数"""
    print("=" * 60)
    print("VAE抗菌多肽生成流程")
    print("=" * 60)
    
    # 1. 训练VAE模型
    print("步骤1: 训练VAE模型")
    vae_model = train_vae_on_peptide_data()
    
    # 2. 加载预测器（可选）
    predictor = None
    try:
        from predict_peptide import PhysChemSeqEngBioBERTPredictor
        predictor = PhysChemSeqEngBioBERTPredictor(
            model_path='results/physchem_seqeng_biobert_dl_rawseq/best_physchem_seqeng_biobert_rawseq_classification.h5',
            scaler_path='results/physchem_seqeng_biobert_dl_rawseq/best_classification_scaler.pkl'
        )
        print("步骤2: 预测器加载成功")
    except Exception as e:
        print(f"步骤2: 预测器加载失败，将跳过潜在空间优化: {e}")
    
    # 3. 生成候选序列
    print("步骤3: 生成候选序列")
    candidates_df = generate_vae_candidates(vae_model, predictor)
    
    # 4. 分析生成的序列
    print("\n步骤4: 分析生成序列特征")
    sequences = candidates_df['sequence'].tolist()
    
    # 长度分析
    lengths = [len(seq) for seq in sequences]
    print(f"序列长度分析:")
    print(f"  范围: {min(lengths)} - {max(lengths)}")
    print(f"  平均: {np.mean(lengths):.1f}")
    
    # 氨基酸组成分析
    from collections import Counter
    all_aa = ''.join(sequences)
    aa_counts = Counter(all_aa)
    
    print(f"\n氨基酸组成分析 (Top 10):")
    for aa, count in aa_counts.most_common(10):
        freq = count / len(all_aa)
        print(f"  {aa}: {freq:.3f}")
    
    print(f"\n=" * 60)
    print("VAE生成完成!")
    print(f"候选序列文件: vae_generated_candidates.csv")
    print(f"模型文件: results/peptide_vae_model_*.h5")
    print("=" * 60)
    
    return vae_model, candidates_df

if __name__ == '__main__':
    vae_model, candidates = main()
