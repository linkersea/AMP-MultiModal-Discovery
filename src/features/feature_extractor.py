import pandas as pd
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from tqdm import tqdm

class FeatureExtractor:
    AA_LIST = "ACDEFGHIKLMNPQRSTVWY"

    def extract_features(self, seq):
        seq = seq.upper()
        seq = ''.join([aa for aa in seq if aa in self.AA_LIST])
        features = {}
        # 氨基酸组成
        for aa in self.AA_LIST:
            features[f"AAC_{aa}"] = seq.count(aa) / len(seq) if len(seq) > 0 else 0
        # 二肽组成（可选，短序列信息有限）
        if len(seq) >= 2:
            for aa1 in self.AA_LIST:
                for aa2 in self.AA_LIST:
                    dipeptide = aa1 + aa2
                    features[f"DPC_{dipeptide}"] = seq.count(dipeptide) / (len(seq) - 1)
        else:
            for aa1 in self.AA_LIST:
                for aa2 in self.AA_LIST:
                    features[f"DPC_{aa1+aa2}"] = 0
        # BioPython理化特征
        if len(seq) > 0:
            pa = ProteinAnalysis(seq)
            features["length"] = len(seq)
            features["molecular_weight"] = pa.molecular_weight()
            features["aromaticity"] = pa.aromaticity()
            features["instability_index"] = pa.instability_index()
            features["isoelectric_point"] = pa.isoelectric_point()
            features["gravy"] = pa.gravy()
            helix, turn, sheet = pa.secondary_structure_fraction()
            features["sec_helix"] = helix
            features["sec_turn"] = turn
            features["sec_sheet"] = sheet
        else:
            features["length"] = 0
            features["molecular_weight"] = 0
            features["aromaticity"] = 0
            features["instability_index"] = 0
            features["isoelectric_point"] = 0
            features["gravy"] = 0
            features["sec_helix"] = 0
            features["sec_turn"] = 0
            features["sec_sheet"] = 0
        return features

    @staticmethod
    def extract_rawseq_features(seqs, max_len=20):
        aa_dict = {aa: i+1 for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        seq_ids = []
        for seq in seqs:
            ids = [aa_dict.get(aa, 0) for aa in seq.upper()[:max_len]]
            ids = ids + [0]*(max_len - len(ids)) if len(ids) < max_len else ids[:max_len]
            seq_ids.append(ids)
        return np.array(seq_ids, dtype=np.int32)

    @staticmethod
    def extract_ngram_features(seqs, n=3):
        from itertools import product
        aa = 'ACDEFGHIKLMNPQRSTVWY'
        ngrams = [''.join(p) for p in product(aa, repeat=n)]
        ngram_idx = {ng: i for i, ng in enumerate(ngrams)}
        features = np.zeros((len(seqs), len(ngrams)), dtype=np.float32)
        for i, seq in enumerate(seqs):
            seq = seq.upper()
            for j in range(len(seq)-n+1):
                ng = seq[j:j+n]
                if ng in ngram_idx:
                    features[i, ngram_idx[ng]] += 1
            if len(seq)-n+1 > 0:
                features[i] /= (len(seq)-n+1)
        return features, ngrams

    @staticmethod
    def extract_window_aac(seqs, window=5):
        aa = 'ACDEFGHIKLMNPQRSTVWY'
        features = []
        for seq in seqs:
            seq = seq.upper()
            win_feats = []
            for i in range(0, len(seq)-window+1):
                win = seq[i:i+window]
                counts = [win.count(a)/window for a in aa]
                win_feats.extend(counts)
            # padding for short seq
            if len(win_feats) < (len(seq)-window+1)*20:
                win_feats += [0]*((len(seq)-window+1)*20 - len(win_feats))
            features.append(win_feats)
        maxlen = max(len(f) for f in features)
        features = [f + [0]*(maxlen-len(f)) for f in features]
        return np.array(features)

    @staticmethod
    def extract_terminal_features(seqs, n=3):
        aa = 'ACDEFGHIKLMNPQRSTVWY'
        features = []
        for seq in seqs:
            seq = seq.upper()
            n_term = seq[:n]
            c_term = seq[-n:]
            n_counts = [n_term.count(a)/n if n else 0 for a in aa]
            c_counts = [c_term.count(a)/n if n else 0 for a in aa]
            features.append(n_counts + c_counts)
        return np.array(features)

    @staticmethod
    def extract_pssm_features(seqs, pssm_dir=None):
        # 伪代码：假设pssm_dir下有与seqs顺序对应的PSSM文件（如iFeature格式）
        # 实际应用需根据PSSM文件格式解析
        # 这里只返回全零特征作为占位
        features = np.zeros((len(seqs), 20))
        return features

    def batch_extract_and_save(self, csv_path, seq_col='sequence', save_path=None):
        df = pd.read_csv(csv_path)
        feature_list = []
        for seq in tqdm(df[seq_col], desc="提取理化特征"):
            try:
                feats = self.extract_features(seq)
            except Exception as e:
                print(f"序列特征提取失败: {seq}, error: {e}")
                feats = {}
            feature_list.append(feats)
        feature_df = pd.DataFrame(feature_list)
        # 合并到原始数据
        df = pd.concat([df, feature_df], axis=1)
        if save_path is None:
            save_path = csv_path
        df.to_csv(save_path, index=False)
        print(f"已保存带理化特征的数据到: {save_path}")