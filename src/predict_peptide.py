"""
PhysChemSeqEngBioBERT+RawSeq CNN模型预测脚本
用法: python src/predict_peptide.py --model_path results/physchem_seqeng_biobert_dl_rawseq/best_physchem_seqeng_biobert_rawseq_classification.h5 --input data/raw/11pep.csv --output pred_result.csv
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.feature_extractor import FeatureExtractor

class PhysChemSeqEngBioBERTPredictor:
    """
    PhysChemSeqEngBioBERT+RawSeq CNN模型预测器
    支持使用训练好的模型对新的多肽序列进行预测
    """
    
    def __init__(self, model_path, scaler_path=None, ngram_n=3, window_size=5, terminal_n=3, rawseq_maxlen=20):
        self.model = keras.models.load_model(model_path)
        
        # 自动查找标准化器文件
        if scaler_path is None:
            model_dir = os.path.dirname(model_path)
            found_scaler_path = os.path.join(model_dir, 'best_classification_scaler.pkl')
            if not os.path.exists(found_scaler_path):
                raise FileNotFoundError(f"未找到标准化器文件: {found_scaler_path}，请在初始化时明确提供 scaler_path")
            scaler_path = found_scaler_path
            
        self.scaler = joblib.load(scaler_path)
        self.feature_extractor = FeatureExtractor()
        
        # 模型参数，与训练时保持一致
        self.ngram_n = ngram_n
        self.window_size = window_size
        self.terminal_n = terminal_n
        self.rawseq_maxlen = rawseq_maxlen
        
        # 氨基酸词典
        self.aa_dict = {aa: i+1 for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        self.vocab_size = len(self.aa_dict) + 1

        # 预加载BioBERT模型
        self.biobert_tokenizer = None
        self.biobert_model = None
        self.load_biobert()
        
        print(f"模型加载完成: {model_path}")
        print(f"标准化器加载完成: {scaler_path}")

    def load_biobert(self):
        """预加载BioBERT分词器和模型"""
        print("正在预加载BioBERT模型...")
        try:
            from transformers import AutoTokenizer, AutoModel
            self.biobert_tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
            self.biobert_model = AutoModel.from_pretrained('dmis-lab/biobert-v1.1')
            self.biobert_model.eval()
            print("BioBERT模型预加载完成。")
        except ImportError:
            print("警告: transformers库未安装，BioBERT嵌入将不可用，将使用随机向量代替。")
        except Exception as e:
            print(f"加载BioBERT模型时出错: {e}。将使用随机向量代替。")
    
    def encode_raw_sequence(self, sequences, max_len=20):
        """编码原始序列为数值数组 - 与训练脚本完全一致"""
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
        """提取理化特征 - 与训练脚本完全一致"""
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
        """提取序列工程特征（包含基础序列特征，与训练脚本完全一致，window AAC特征补齐到训练时的真实维度）"""
        print("提取序列工程特征...")
        # Basic sequence features
        seq_features = []
        for seq in df['sequence']:
            features = self.feature_extractor.extract_features(seq)
            seq_features.append(list(features.values()))
        seq_features = np.array(seq_features)
        # N-gram features
        ngram_features, ngram_names = self.feature_extractor.extract_ngram_features(
            df['sequence'], n=self.ngram_n
        )
        # Window AAC features  
        window_aac_features = self.feature_extractor.extract_window_aac(
            df['sequence'], window=self.window_size
        )
        # 读取训练时window AAC特征真实维度
        window_aac_dim_path = os.path.join(os.path.dirname(__file__), '../results/physchem_seqeng_biobert_dl_rawseq/window_aac_dim.npy')
        window_aac_dim = int(np.load(window_aac_dim_path)[0])
        if window_aac_features.shape[1] < window_aac_dim:
            pad_width = window_aac_dim - window_aac_features.shape[1]
            window_aac_features = np.pad(window_aac_features, ((0,0),(0,pad_width)), 'constant')
        elif window_aac_features.shape[1] > window_aac_dim:
            window_aac_features = window_aac_features[:, :window_aac_dim]
        # Terminal features
        terminal_features = self.feature_extractor.extract_terminal_features(
            df['sequence'], n=self.terminal_n
        )
        # Combine all sequence engineering features
        all_features = np.concatenate([
            seq_features, ngram_features, window_aac_features, terminal_features
        ], axis=1)
        # 生成特征名
        seq_feature_names = list(self.feature_extractor.extract_features(df['sequence'].iloc[0]).keys())
        ngram_feature_names = [f"ngram_{name}" for name in ngram_names]
        window_aac_names = [f"winAAC_{i+1}" for i in range(window_aac_features.shape[1])]
        terminal_names = ([f"Nterm_{aa}" for aa in self.feature_extractor.AA_LIST] + 
                         [f"Cterm_{aa}" for aa in self.feature_extractor.AA_LIST])
        all_feature_names = seq_feature_names + ngram_feature_names + window_aac_names + terminal_names
        return all_features, all_feature_names
    
    def generate_biobert_embeddings(self, sequences):
        """使用预加载的BioBERT模型生成嵌入"""
        print("生成BioBERT嵌入...")
        
        # 检查BioBERT模型是否已成功加载
        if self.biobert_model is None or self.biobert_tokenizer is None:
            print("警告: BioBERT模型未加载，使用随机嵌入代替BioBERT。")
            embeddings = np.random.randn(len(sequences), 768)  # 768 is BioBERT embedding size
            print(f"随机嵌入生成完成，维度: {embeddings.shape}")
            return embeddings

        try:
            import torch
            
            embeddings = []
            print("使用预加载的BioBERT模型生成序列嵌入...")
            with torch.no_grad():
                for seq in sequences:
                    # 为BioBERT在氨基酸之间添加空格
                    spaced_seq = ' '.join(list(seq))
                    inputs = self.biobert_tokenizer(spaced_seq, return_tensors="pt", 
                                     truncation=True, max_length=512, padding=True)
                    outputs = self.biobert_model(**inputs)
                    # 使用[CLS]标记的嵌入
                    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                    embeddings.append(cls_embedding)
            
            embeddings = np.array(embeddings)
            print(f"BioBERT嵌入生成完成，维度: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            print(f"生成BioBERT嵌入时发生错误: {e}")
            print("将使用随机嵌入作为备选方案。")
            embeddings = np.random.randn(len(sequences), 768)
            print(f"随机嵌入生成完成，维度: {embeddings.shape}")
            return embeddings
    
    def load_and_prepare_features(self, df):
        """加载和准备所有特征 - 与训练脚本完全一致"""
        print("准备physchem_seqeng_biobert_dl_rawseq特征组合...")
        
        # 1. Physico-chemical features
        physchem_features, physchem_names = self.extract_physico_chemical_features(df)
        
        # 2. Sequence engineering features
        seqeng_features, seqeng_names = self.extract_sequence_engineering_features(df)
        
        # 3. BioBERT embeddings
        biobert_features = self.generate_biobert_embeddings(df['sequence'])
        biobert_names = [f"biobert_{i}" for i in range(biobert_features.shape[1])]
        
        # 4. Raw sequence features (for CNN)
        rawseq_features = self.encode_raw_sequence(df['sequence'], max_len=self.rawseq_maxlen)
        
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
    
    def predict(self, sequences):
        """对序列进行预测"""
        print("开始预测...")
        
        # 创建DataFrame格式，与训练时一致
        df = pd.DataFrame({'sequence': sequences})
        
        # 准备特征
        traditional_features, rawseq_features, _ = self.load_and_prepare_features(df)
        
        # 标准化传统特征
        traditional_features_scaled = self.scaler.transform(traditional_features)
        
        # 模型预测
        y_prob = self.model.predict([rawseq_features, traditional_features_scaled]).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        
        return y_prob, y_pred

def main():
    parser = argparse.ArgumentParser(description='PhysChemSeqEngBioBERT+RawSeq CNN模型预测')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='训练好的模型路径 (.h5文件)')
    parser.add_argument('--scaler_path', type=str, 
                       help='标准化器路径 (.pkl文件)，如果不提供则自动查找')
    parser.add_argument('--input', type=str, required=True, 
                       help='输入CSV文件，需包含sequence列')
    parser.add_argument('--output', type=str, required=True, 
                       help='输出CSV文件路径')
    parser.add_argument('--rawseq_maxlen', type=int, default=20,
                       help='原始序列最大长度')
    parser.add_argument('--ngram_n', type=int, default=3,
                       help='N-gram特征的N值')
    parser.add_argument('--window_size', type=int, default=5,
                       help='窗口AAC特征窗口大小')
    parser.add_argument('--terminal_n', type=int, default=3,
                       help='末端特征长度')
    
    args = parser.parse_args()
    
    # 初始化预测器
    predictor = PhysChemSeqEngBioBERTPredictor(
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        ngram_n=args.ngram_n,
        window_size=args.window_size,
        terminal_n=args.terminal_n,
        rawseq_maxlen=args.rawseq_maxlen
    )
    
    print("="*60)
    print("PhysChemSeqEngBioBERT+RawSeq CNN预测")
    print("="*60)
    print(f"模型路径: {predictor.model.name}")
    print(f"标准化器路径: {args.scaler_path or '自动查找'}")
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print("="*60)
    
    # 加载数据
    df = pd.read_csv(args.input)
    if 'sequence' not in df.columns:
        raise ValueError('输入CSV文件必须包含sequence列')
    
    print(f"加载了{len(df)}个序列")
    print("序列列表:")
    for i, seq in enumerate(df['sequence'], 1):
        print(f"  {i}. {seq}")
    
    # 进行预测
    y_prob, y_pred = predictor.predict(df['sequence'].tolist())
    
    # 保存结果
    df['pred_probability'] = y_prob
    df['pred_label'] = y_pred
    df['pred_activity'] = ['高活性' if label == 1 else '低活性' for label in y_pred]
    
    df.to_csv(args.output, index=False)
    
    print("\n" + "="*60)
    print("预测结果:")
    print("="*60)
    for i, row in df.iterrows():
        print(f"{i+1}. {row['sequence']}")
        print(f"   预测概率: {row['pred_probability']:.4f}")
        print(f"   预测标签: {row['pred_label']} ({row['pred_activity']})")
        print()
    
    print(f"预测完成！结果已保存到: {args.output}")

if __name__ == '__main__':
    main()
