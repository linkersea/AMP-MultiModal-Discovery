#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive training script for physchem_seqeng_biobert_dl_rawseq feature combination
with RawSeq_CNN model using 5-fold cross-validation for both classification and regression tasks.

This script implements the best-performing configuration identified from previous experiments:
- Feature input: physchem_seqeng_biobert_dl_rawseq (physico-chemical + sequence engineering + BioBERT + raw sequence)
- Model: RawSeq_CNN (1D CNN for raw sequence processing)
- Validation: 5-fold cross-validation
- Tasks: Both binary classification and regression
- Model saving: Best models saved for future use
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras
from keras import layers
import joblib
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = ['SimHei', 'Arial', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.feature_extractor import FeatureExtractor

class PhysChemSeqEngBioBERTRawSeqPipeline:
    """
    Comprehensive pipeline for physchem_seqeng_biobert_dl_rawseq feature combination
    """
    
    def __init__(self, args):
        self.args = args
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        
        # Amino acid vocabulary for raw sequence encoding
        self.aa_dict = {aa: i+1 for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        self.vocab_size = len(self.aa_dict) + 1  # +1 for padding
        
    def encode_raw_sequence(self, sequences, max_len=20):
        """Encode raw sequences to numerical arrays"""
        encoded_seqs = []
        for seq in sequences:
            # Convert to uppercase and keep only valid amino acids
            seq_clean = ''.join([aa for aa in seq.upper() if aa in self.aa_dict])
            # Encode to numbers
            encoded = [self.aa_dict.get(aa, 0) for aa in seq_clean[:max_len]]
            # Pad or truncate to max_len
            if len(encoded) < max_len:
                encoded = encoded + [0] * (max_len - len(encoded))
            else:
                encoded = encoded[:max_len]
            encoded_seqs.append(encoded)
        return np.array(encoded_seqs, dtype=np.int32)
    
    def extract_physico_chemical_features(self, df):
        """Extract physico-chemical features"""
        print("提取理化特征...")
        physchem_features = []
        feature_names = []
        
        for seq in df['sequence']:
            features = self.feature_extractor.extract_features(seq)
            physchem_features.append(list(features.values()))
            if not feature_names:
                feature_names = list(features.keys())
        
        return np.array(physchem_features), feature_names
    
    def extract_sequence_engineering_features(self, df):
        """Extract sequence engineering features including n-grams, window AAC, terminal features"""
        print("提取序列工程特征...")
        
        # Basic sequence features
        seq_features = []
        for seq in df['sequence']:
            features = self.feature_extractor.extract_features(seq)
            seq_features.append(list(features.values()))
        seq_features = np.array(seq_features)
        
        # N-gram features
        ngram_features, ngram_names = self.feature_extractor.extract_ngram_features(
            df['sequence'], n=self.args.ngram_n
        )
        
        # Window AAC features  
        window_aac_features = self.feature_extractor.extract_window_aac(
            df['sequence'], window=self.args.window_size
        )
        # 保存window AAC特征真实维度（只保存一次即可，重复保存也无影响）
        window_aac_dim_path = os.path.join('results/physchem_seqeng_biobert_dl_rawseq', 'window_aac_dim.npy')
        np.save(window_aac_dim_path, [window_aac_features.shape[1]])
        
        # Terminal features
        terminal_features = self.feature_extractor.extract_terminal_features(
            df['sequence'], n=self.args.terminal_n
        )
        
        # Combine all sequence engineering features
        all_features = np.concatenate([
            seq_features, ngram_features, window_aac_features, terminal_features
        ], axis=1)
        
        # Generate feature names
        seq_feature_names = list(self.feature_extractor.extract_features(df['sequence'].iloc[0]).keys())
        ngram_feature_names = [f"ngram_{name}" for name in ngram_names]
        window_aac_names = [f"winAAC_{i+1}" for i in range(window_aac_features.shape[1])]
        terminal_names = ([f"Nterm_{aa}" for aa in self.feature_extractor.AA_LIST] + 
                         [f"Cterm_{aa}" for aa in self.feature_extractor.AA_LIST])
        
        all_feature_names = seq_feature_names + ngram_feature_names + window_aac_names + terminal_names
        
        return all_features, all_feature_names
    
    def load_biobert_embeddings(self, biobert_path, sequences):
        """Load or generate BioBERT embeddings"""
        if os.path.exists(biobert_path):
            print(f"加载BioBERT嵌入: {biobert_path}")
            embeddings = np.load(biobert_path)
        else:
            print("生成BioBERT嵌入...")
            embeddings = self.generate_biobert_embeddings(sequences, biobert_path)
        
        return embeddings
    
    def generate_biobert_embeddings(self, sequences, save_path):
        """Generate BioBERT embeddings if not exist"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            print("加载BioBERT模型...")
            tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
            model = AutoModel.from_pretrained('dmis-lab/biobert-v1.1')
            model.eval()
            
            embeddings = []
            print("生成序列嵌入...")
            with torch.no_grad():
                for seq in sequences:
                    # Add spaces between amino acids for BioBERT
                    spaced_seq = ' '.join(list(seq))
                    inputs = tokenizer(spaced_seq, return_tensors="pt", 
                                     truncation=True, max_length=512, padding=True)
                    outputs = model(**inputs)
                    # Use [CLS] token embedding
                    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                    embeddings.append(cls_embedding)
            
            embeddings = np.array(embeddings)
            
            # Save embeddings
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, embeddings)
            print(f"BioBERT嵌入已保存: {save_path}")
            
            return embeddings
            
        except ImportError:
            print("警告: transformers库未安装，使用随机嵌入代替BioBERT")
            # Generate random embeddings as fallback
            embeddings = np.random.randn(len(sequences), 768)  # 768 is BioBERT embedding size
            np.save(save_path, embeddings)
            return embeddings
    
    def load_and_prepare_features(self, df):
        """Load and prepare all features for physchem_seqeng_biobert_dl_rawseq combination"""
        print("准备physchem_seqeng_biobert_dl_rawseq特征组合...")
        
        # 1. Physico-chemical features
        physchem_features, physchem_names = self.extract_physico_chemical_features(df)
        
        # 2. Sequence engineering features
        seqeng_features, seqeng_names = self.extract_sequence_engineering_features(df)
        
        # 3. BioBERT embeddings
        biobert_features = self.load_biobert_embeddings(self.args.biobert_emb, df['sequence'])
        biobert_names = [f"biobert_{i}" for i in range(biobert_features.shape[1])]
        
        # 4. Raw sequence features (for CNN)
        rawseq_features = self.encode_raw_sequence(df['sequence'], max_len=self.args.rawseq_maxlen)
        
        # Combine traditional features (physchem + seqeng + biobert) for dense layers
        traditional_features = np.concatenate([
            physchem_features, seqeng_features, biobert_features
        ], axis=1)
        
        # Feature names
        feature_names = physchem_names + seqeng_names + biobert_names
        
        print(f"特征维度:")
        print(f"  - 理化特征: {physchem_features.shape[1]}")
        print(f"  - 序列工程特征: {seqeng_features.shape[1]}")
        print(f"  - BioBERT特征: {biobert_features.shape[1]}")
        print(f"  - 传统特征总计: {traditional_features.shape[1]}")
        print(f"  - 原始序列特征: {rawseq_features.shape}")
        
        return traditional_features, rawseq_features, feature_names
    
    def build_rawseq_cnn_model(self, traditional_feature_dim, task='classification'):
        """
        Build RawSeq_CNN model that combines CNN for raw sequences with dense layers for traditional features
        """
        # Raw sequence input (for CNN)
        rawseq_input = layers.Input(shape=(self.args.rawseq_maxlen,), name='rawseq_input')
        
        # CNN branch for raw sequences
        x_cnn = layers.Embedding(
            input_dim=self.vocab_size, 
            output_dim=self.args.embedding_dim, 
            input_length=self.args.rawseq_maxlen
        )(rawseq_input)
        x_cnn = layers.Conv1D(64, 3, activation='relu')(x_cnn)
        x_cnn = layers.MaxPooling1D(2)(x_cnn)
        x_cnn = layers.Flatten()(x_cnn)
        
        # Traditional features input (physchem + seqeng + biobert)
        traditional_input = layers.Input(shape=(traditional_feature_dim,), name='traditional_input')
        
        # Dense branch for traditional features
        x_dense = layers.Dense(128, activation='relu')(traditional_input)
        x_dense = layers.Dropout(0.3)(x_dense)
        x_dense = layers.Dense(64, activation='relu')(x_dense)
        
        # Combine CNN and dense branches
        combined = layers.Concatenate()([x_cnn, x_dense])
        combined = layers.Dense(64, activation='relu')(combined)
        combined = layers.Dropout(0.2)(combined)
        
        # Output layer
        if task == 'classification':
            output = layers.Dense(1, activation='sigmoid', name='output')(combined)
            model = keras.Model(inputs=[rawseq_input, traditional_input], outputs=output)
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
        else:  # regression
            output = layers.Dense(1, activation='linear', name='output')(combined)
            model = keras.Model(inputs=[rawseq_input, traditional_input], outputs=output)
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
        
        return model
    
    def perform_classification_cv(self, traditional_features, rawseq_features, y, df):
        """Perform 5-fold cross-validation for classification task"""
        print("\n" + "="*60)
        print("开始分类任务的五折交叉验证")
        print("="*60)
        
        # Use stratified k-fold for classification
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        fold_metrics = []
        all_y_true, all_y_pred, all_y_prob = [], [], []
        best_model = None
        best_auc = -1
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(traditional_features, y)):
            print(f"\n--- Fold {fold + 1}/5 ---")
            
            # Split data
            X_trad_train, X_trad_val = traditional_features[train_idx], traditional_features[val_idx]
            X_raw_train, X_raw_val = rawseq_features[train_idx], rawseq_features[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale traditional features
            scaler = StandardScaler()
            X_trad_train_scaled = scaler.fit_transform(X_trad_train)
            X_trad_val_scaled = scaler.transform(X_trad_val)
            
            # Build and train model
            model = self.build_rawseq_cnn_model(X_trad_train_scaled.shape[1], task='classification')
            
            # Callbacks
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            # Train
            history = model.fit(
                [X_raw_train, X_trad_train_scaled], y_train,
                validation_data=([X_raw_val, X_trad_val_scaled], y_val),
                epochs=self.args.epochs,
                batch_size=self.args.batch_size,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Predict
            y_prob = model.predict([X_raw_val, X_trad_val_scaled]).flatten()
            y_pred = (y_prob > 0.5).astype(int)
            
            # Calculate metrics
            acc = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_prob)
            f1 = f1_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            
            fold_metrics.append({
                'fold': fold + 1,
                'accuracy': acc,
                'auc': auc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            })
            
            # Store predictions for overall evaluation
            all_y_true.extend(y_val)
            all_y_pred.extend(y_pred)
            all_y_prob.extend(y_prob)
            
            # Save best model
            if auc > best_auc:
                best_auc = auc
                best_model = model
                # Save scaler for best model
                joblib.dump(scaler, os.path.join(self.args.out_dir, 'best_classification_scaler.pkl'))
            
            print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        # Calculate overall metrics
        overall_acc = accuracy_score(all_y_true, all_y_pred)
        overall_auc = roc_auc_score(all_y_true, all_y_prob)
        overall_f1 = f1_score(all_y_true, all_y_pred)
        overall_precision = precision_score(all_y_true, all_y_pred)
        overall_recall = recall_score(all_y_true, all_y_pred)
        
        # Print summary
        print(f"\n{'='*60}")
        print("分类任务交叉验证结果总结:")
        print(f"{'='*60}")
        metrics_df = pd.DataFrame(fold_metrics)
        print(metrics_df.round(4))
        print(f"\n平均指标:")
        print(f"Accuracy: {metrics_df['accuracy'].mean():.4f} ± {metrics_df['accuracy'].std():.4f}")
        print(f"AUC: {metrics_df['auc'].mean():.4f} ± {metrics_df['auc'].std():.4f}")
        print(f"F1: {metrics_df['f1'].mean():.4f} ± {metrics_df['f1'].std():.4f}")
        print(f"Precision: {metrics_df['precision'].mean():.4f} ± {metrics_df['precision'].std():.4f}")
        print(f"Recall: {metrics_df['recall'].mean():.4f} ± {metrics_df['recall'].std():.4f}")
        
        print(f"\n整体指标 (所有fold合并):")
        print(f"Overall Accuracy: {overall_acc:.4f}")
        print(f"Overall AUC: {overall_auc:.4f}")
        print(f"Overall F1: {overall_f1:.4f}")
        print(f"Overall Precision: {overall_precision:.4f}")
        print(f"Overall Recall: {overall_recall:.4f}")
        
        # Save results
        metrics_df.to_csv(os.path.join(self.args.out_dir, 'classification_cv_results.csv'), index=False)
        
        # Save best model
        best_model.save(os.path.join(self.args.out_dir, 'best_physchem_seqeng_biobert_rawseq_classification.h5'))
        print(f"\n最佳分类模型已保存: {os.path.join(self.args.out_dir, 'best_physchem_seqeng_biobert_rawseq_classification.h5')}")
        
        # Generate visualizations
        self.plot_classification_results(all_y_true, all_y_pred, all_y_prob, metrics_df)
        
        return metrics_df, best_model
    
    def perform_regression_cv(self, traditional_features, rawseq_features, y, df):
        """Perform 5-fold cross-validation for regression task"""
        print("\n" + "="*60)
        print("开始回归任务的五折交叉验证")
        print("="*60)
        
        # Use regular k-fold for regression
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        fold_metrics = []
        all_y_true, all_y_pred = [], []
        best_model = None
        best_r2 = -np.inf
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(traditional_features)):
            print(f"\n--- Fold {fold + 1}/5 ---")
            
            # Split data
            X_trad_train, X_trad_val = traditional_features[train_idx], traditional_features[val_idx]
            X_raw_train, X_raw_val = rawseq_features[train_idx], rawseq_features[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale traditional features
            scaler = StandardScaler()
            X_trad_train_scaled = scaler.fit_transform(X_trad_train)
            X_trad_val_scaled = scaler.transform(X_trad_val)
            
            # Build and train model
            model = self.build_rawseq_cnn_model(X_trad_train_scaled.shape[1], task='regression')
            
            # Callbacks
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            # Train
            history = model.fit(
                [X_raw_train, X_trad_train_scaled], y_train,
                validation_data=([X_raw_val, X_trad_val_scaled], y_val),
                epochs=self.args.epochs,
                batch_size=self.args.batch_size,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Predict
            y_pred = model.predict([X_raw_val, X_trad_val_scaled]).flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mse)
            
            fold_metrics.append({
                'fold': fold + 1,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            })
            
            # Store predictions for overall evaluation
            all_y_true.extend(y_val)
            all_y_pred.extend(y_pred)
            
            # Save best model
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                # Save scaler for best model
                joblib.dump(scaler, os.path.join(self.args.out_dir, 'best_regression_scaler.pkl'))
            
            print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Calculate overall metrics
        overall_mse = mean_squared_error(all_y_true, all_y_pred)
        overall_mae = mean_absolute_error(all_y_true, all_y_pred)
        overall_r2 = r2_score(all_y_true, all_y_pred)
        overall_rmse = np.sqrt(overall_mse)
        
        # Print summary
        print(f"\n{'='*60}")
        print("回归任务交叉验证结果总结:")
        print(f"{'='*60}")
        metrics_df = pd.DataFrame(fold_metrics)
        print(metrics_df.round(4))
        print(f"\n平均指标:")
        print(f"MSE: {metrics_df['mse'].mean():.4f} ± {metrics_df['mse'].std():.4f}")
        print(f"RMSE: {metrics_df['rmse'].mean():.4f} ± {metrics_df['rmse'].std():.4f}")
        print(f"MAE: {metrics_df['mae'].mean():.4f} ± {metrics_df['mae'].std():.4f}")
        print(f"R²: {metrics_df['r2'].mean():.4f} ± {metrics_df['r2'].std():.4f}")
        
        print(f"\n整体指标 (所有fold合并):")
        print(f"Overall MSE: {overall_mse:.4f}")
        print(f"Overall RMSE: {overall_rmse:.4f}")
        print(f"Overall MAE: {overall_mae:.4f}")
        print(f"Overall R²: {overall_r2:.4f}")
        
        # Save results
        metrics_df.to_csv(os.path.join(self.args.out_dir, 'regression_cv_results.csv'), index=False)
        
        # Save best model
        best_model.save(os.path.join(self.args.out_dir, 'best_physchem_seqeng_biobert_rawseq_regression.h5'))
        print(f"\n最佳回归模型已保存: {os.path.join(self.args.out_dir, 'best_physchem_seqeng_biobert_rawseq_regression.h5')}")
        
        # Generate visualizations
        self.plot_regression_results(all_y_true, all_y_pred, metrics_df)
        
        return metrics_df, best_model
    
    def plot_classification_results(self, y_true, y_pred, y_prob, metrics_df):
        """Generate classification visualization plots"""
        print("\n生成分类任务可视化图表...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PhysChemSeqEngBioBERT+RawSeq CNN 分类任务结果', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('混淆矩阵')
        axes[0,0].set_xlabel('预测标签')
        axes[0,0].set_ylabel('真实标签')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        axes[0,1].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_score:.3f}')
        axes[0,1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0,1].set_xlabel('假正率 (FPR)')
        axes[0,1].set_ylabel('真正率 (TPR)')
        axes[0,1].set_title('ROC曲线')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        axes[0,2].plot(recall, precision, linewidth=2)
        axes[0,2].set_xlabel('召回率')
        axes[0,2].set_ylabel('精确率')
        axes[0,2].set_title('精确率-召回率曲线')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Metrics by Fold
        metrics_to_plot = ['accuracy', 'auc', 'f1']
        x_pos = np.arange(len(metrics_df))
        width = 0.25
        
        for i, metric in enumerate(metrics_to_plot):
            axes[1,0].bar(x_pos + i*width, metrics_df[metric], width, 
                         label=metric.upper(), alpha=0.8)
        
        axes[1,0].set_xlabel('Fold')
        axes[1,0].set_ylabel('分数')
        axes[1,0].set_title('各折指标对比')
        axes[1,0].set_xticks(x_pos + width)
        axes[1,0].set_xticklabels([f'Fold {i+1}' for i in range(len(metrics_df))])
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Distribution of Predictions
        axes[1,1].hist([np.array(y_prob)[np.array(y_true)==0], 
                        np.array(y_prob)[np.array(y_true)==1]], 
                       bins=30, alpha=0.7, label=['负样本', '正样本'])
        axes[1,1].set_xlabel('预测概率')
        axes[1,1].set_ylabel('频次')
        axes[1,1].set_title('预测概率分布')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Cross-validation metrics statistics
        metrics_stats = metrics_df[['accuracy', 'auc', 'f1', 'precision', 'recall']].describe()
        axes[1,2].axis('off')
        table_data = []
        for metric in ['accuracy', 'auc', 'f1', 'precision', 'recall']:
            mean_val = metrics_stats.loc['mean', metric]
            std_val = metrics_stats.loc['std', metric]
            table_data.append([metric.upper(), f'{mean_val:.4f}', f'{std_val:.4f}'])
        
        table = axes[1,2].table(cellText=table_data,
                               colLabels=['指标', '均值', '标准差'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1,2].set_title('交叉验证统计结果')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.out_dir, 'classification_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"分类结果图表已保存: {os.path.join(self.args.out_dir, 'classification_results.png')}")
    
    def plot_regression_results(self, y_true, y_pred, metrics_df):
        """Generate regression visualization plots"""
        print("\n生成回归任务可视化图表...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PhysChemSeqEngBioBERT+RawSeq CNN 回归任务结果', fontsize=16, fontweight='bold')
        
        # 1. True vs Predicted scatter plot
        axes[0,0].scatter(y_true, y_pred, alpha=0.6, s=30)
        axes[0,0].plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', linewidth=2)
        axes[0,0].set_xlabel('真实值')
        axes[0,0].set_ylabel('预测值')
        axes[0,0].set_title(f'真实值 vs 预测值\nR² = {r2_score(y_true, y_pred):.4f}')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Residuals plot
        residuals = np.array(y_pred) - np.array(y_true)
        axes[0,1].scatter(y_pred, residuals, alpha=0.6, s=30)
        axes[0,1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0,1].set_xlabel('预测值')
        axes[0,1].set_ylabel('残差')
        axes[0,1].set_title('残差图')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Distribution of residuals
        axes[0,2].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[0,2].set_xlabel('残差')
        axes[0,2].set_ylabel('频次')
        axes[0,2].set_title('残差分布')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Metrics by Fold
        metrics_to_plot = ['r2', 'mse', 'mae']
        x_pos = np.arange(len(metrics_df))
        width = 0.25
        
        for i, metric in enumerate(metrics_to_plot):
            if metric == 'mse':
                # Scale MSE for better visualization
                scaled_values = metrics_df[metric] / metrics_df[metric].max()
                axes[1,0].bar(x_pos + i*width, scaled_values, width, 
                             label=f'{metric.upper()} (scaled)', alpha=0.8)
            else:
                axes[1,0].bar(x_pos + i*width, metrics_df[metric], width, 
                             label=metric.upper(), alpha=0.8)
        
        axes[1,0].set_xlabel('Fold')
        axes[1,0].set_ylabel('分数')
        axes[1,0].set_title('各折指标对比')
        axes[1,0].set_xticks(x_pos + width)
        axes[1,0].set_xticklabels([f'Fold {i+1}' for i in range(len(metrics_df))])
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Error distribution
        errors = np.abs(residuals)
        axes[1,1].hist(errors, bins=30, alpha=0.7, edgecolor='black')
        axes[1,1].set_xlabel('绝对误差')
        axes[1,1].set_ylabel('频次')
        axes[1,1].set_title('绝对误差分布')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Cross-validation metrics statistics
        metrics_stats = metrics_df[['mse', 'rmse', 'mae', 'r2']].describe()
        axes[1,2].axis('off')
        table_data = []
        for metric in ['mse', 'rmse', 'mae', 'r2']:
            mean_val = metrics_stats.loc['mean', metric]
            std_val = metrics_stats.loc['std', metric]
            table_data.append([metric.upper(), f'{mean_val:.4f}', f'{std_val:.4f}'])
        
        table = axes[1,2].table(cellText=table_data,
                               colLabels=['指标', '均值', '标准差'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1,2].set_title('交叉验证统计结果')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.out_dir, 'regression_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"回归结果图表已保存: {os.path.join(self.args.out_dir, 'regression_results.png')}")

def main():
    parser = argparse.ArgumentParser(description='PhysChemSeqEngBioBERT+RawSeq CNN 训练管道')
    
    # Data paths
    parser.add_argument('--raw_csv', type=str, default='data/raw/120dataset.csv',
                       help='原始数据CSV文件路径')
    parser.add_argument('--physchem_csv', type=str, default='data/processed/120dataset_physchem.csv',
                       help='理化特征CSV文件路径')
    parser.add_argument('--biobert_emb', type=str, default='data/processed/biobert_emb.npy',
                       help='BioBERT嵌入文件路径')
    parser.add_argument('--out_dir', type=str, default='results/physchem_seqeng_biobert_dl_rawseq',
                       help='输出目录')
    
    # Feature extraction parameters
    parser.add_argument('--rawseq_maxlen', type=int, default=20,
                       help='原始序列最大长度')
    parser.add_argument('--embedding_dim', type=int, default=32,
                       help='嵌入层维度')
    parser.add_argument('--ngram_n', type=int, default=3,
                       help='N-gram特征的N值')
    parser.add_argument('--window_size', type=int, default=5,
                       help='窗口AAC特征窗口大小')
    parser.add_argument('--terminal_n', type=int, default=3,
                       help='末端特征长度')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--task', type=str, choices=['classification', 'regression', 'both'], 
                       default='both', help='任务类型')
    parser.add_argument('--classify_threshold', type=float, default=0.7,
                       help='分类任务的活性阈值')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save arguments
    import json
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("="*80)
    print("PhysChemSeqEngBioBERT+RawSeq CNN 综合训练管道")
    print("="*80)
    print(f"输出目录: {args.out_dir}")
    print(f"任务类型: {args.task}")
    print("="*80)
    
    # Initialize pipeline
    pipeline = PhysChemSeqEngBioBERTRawSeqPipeline(args)
    
    # Load data
    print("加载数据...")
    df = pd.read_csv(args.raw_csv)
    print(f"数据形状: {df.shape}")
    print(f"活性范围: {df['activity'].min():.4f} - {df['activity'].max():.4f}")
    
    # Load physico-chemical features if available
    if os.path.exists(args.physchem_csv):
        print("合并理化特征...")
        df_physchem = pd.read_csv(args.physchem_csv)
        df = df.merge(df_physchem, on=['sequence', 'activity'], how='left')
        print(f"合并后数据形状: {df.shape}")
    
    # Prepare features
    traditional_features, rawseq_features, feature_names = pipeline.load_and_prepare_features(df)
    
    # Run tasks
    if args.task in ['classification', 'both']:
        # Classification task
        y_class = (df['activity'] >= args.classify_threshold).astype(int).values
        print(f"\n分类标签分布: 负样本={np.sum(y_class==0)}, 正样本={np.sum(y_class==1)}")
        
        classification_results, best_classification_model = pipeline.perform_classification_cv(
            traditional_features, rawseq_features, y_class, df
        )
    
    if args.task in ['regression', 'both']:
        # Regression task
        y_reg = df['activity'].values
        print(f"\n回归目标统计: 均值={np.mean(y_reg):.4f}, 标准差={np.std(y_reg):.4f}")
        
        regression_results, best_regression_model = pipeline.perform_regression_cv(
            traditional_features, rawseq_features, y_reg, df
        )
    
    print("\n" + "="*80)
    print("训练完成！")
    print("="*80)
    print(f"所有结果已保存到: {args.out_dir}")
    print("包含文件:")
    for file in os.listdir(args.out_dir):
        print(f"  - {file}")

if __name__ == '__main__':
    main()
