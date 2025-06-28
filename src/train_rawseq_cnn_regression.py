import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
from features.feature_extractor import FeatureExtractor

def extract_rawseq_features(seqs, max_len=20):
    aa_dict = {aa: i+1 for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    seq_ids = []
    for seq in seqs:
        ids = [aa_dict.get(aa, 0) for aa in seq.upper()[:max_len]]
        ids = ids + [0]*(max_len - len(ids)) if len(ids) < max_len else ids[:max_len]
        seq_ids.append(ids)
    return np.array(seq_ids, dtype=np.int32)

def load_features(df, biobert_emb_path, args):
    features = []
    feature_names = []
    fe = FeatureExtractor()
    # physchem
    physchem_cols = [col for col in df.columns if col not in ['sequence', 'activity', 'label']]
    features.append(df[physchem_cols].values)
    feature_names += physchem_cols
    # seqeng
    seq_features = np.array([list(fe.extract_features(seq).values()) for seq in df['sequence']])
    features.append(seq_features)
    feature_names += list(fe.extract_features(df['sequence'].iloc[0]).keys())
    ngram_features, ngram_names = fe.extract_ngram_features(df['sequence'], n=args.ngram_n)
    features.append(ngram_features)
    feature_names += [f"ngram_{ng}" for ng in ngram_names]
    window_aac_features = fe.extract_window_aac(df['sequence'], window=args.window_size)
    features.append(window_aac_features)
    feature_names += [f"winAAC_{i+1}" for i in range(window_aac_features.shape[1])]
    terminal_features = fe.extract_terminal_features(df['sequence'], n=args.terminal_n)
    features.append(terminal_features)
    feature_names += [f"Nterm_{aa}" for aa in fe.AA_LIST] + [f"Cterm_{aa}" for aa in fe.AA_LIST]
    # biobert
    biobert_emb = np.load(biobert_emb_path)
    features.append(biobert_emb)
    feature_names += [f"biobert_{i}" for i in range(biobert_emb.shape[1])]
    # rawseq
    rawseq_features = fe.extract_rawseq_features(df['sequence'], max_len=args.rawseq_maxlen)
    features.append(rawseq_features)
    feature_names += [f"pos_{i+1}" for i in range(rawseq_features.shape[1])]
    X = np.concatenate(features, axis=1) if len(features) > 1 else features[0]
    return X, feature_names, rawseq_features

def build_rawseq_cnn_regression(rawseq_maxlen=20, vocab_size=21, embedding_dim=32):
    model = keras.Sequential([
        keras.layers.Input(shape=(rawseq_maxlen,)),
        keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=rawseq_maxlen),
        keras.layers.Conv1D(64, 3, activation='relu'),
        keras.layers.MaxPooling1D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_rawseq_cnn_classification(rawseq_maxlen=20, vocab_size=21, embedding_dim=32):
    model = keras.Sequential([
        keras.layers.Input(shape=(rawseq_maxlen,)),
        keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=rawseq_maxlen),
        keras.layers.Conv1D(64, 3, activation='relu'),
        keras.layers.MaxPooling1D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_csv', type=str, default='data/raw/120dataset.csv')
    parser.add_argument('--physchem_csv', type=str, default='data/processed/120dataset_physchem.csv')
    parser.add_argument('--biobert_emb', type=str, default='data/processed/biobert_emb.npy')
    parser.add_argument('--out_dir', type=str, default='results/regression')
    parser.add_argument('--rawseq_maxlen', type=int, default=20)
    parser.add_argument('--ngram_n', type=int, default=3)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--terminal_n', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--task', type=str, choices=['regression', 'classification'], default='regression', help='任务类型')
    parser.add_argument('--classify_threshold', type=float, default=0.7, help='分类任务的活性阈值')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.raw_csv)
    if not os.path.exists(args.physchem_csv):
        fe = FeatureExtractor()
        fe.batch_extract_and_save(args.raw_csv, seq_col='sequence', save_path=args.physchem_csv)
    df_physchem = pd.read_csv(args.physchem_csv)
    df = df.merge(df_physchem, on=['sequence', 'activity'], how='left')
    X, feature_names, rawseq_features = load_features(df, args.biobert_emb, args)
    X_rawseq = rawseq_features

    # 检查biobert嵌入文件是否存在，不存在则自动提取
    if not os.path.exists(args.biobert_emb):
        from transformers import AutoTokenizer, AutoModel
        import torch
        print(f"[INFO] 提取BioBERT嵌入: {args.biobert_emb}")
        tokenizer = AutoTokenizer.from_pretrained('model/biobert/biobert')
        model = AutoModel.from_pretrained('model/biobert/biobert')
        model.eval()
        embeddings = []
        with torch.no_grad():
            for seq in df['sequence']:
                inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=512)
                outputs = model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                embeddings.append(cls_emb)
        embeddings = np.array(embeddings)
        np.save(args.biobert_emb, embeddings)
        print(f"[INFO] BioBERT嵌入已保存到: {args.biobert_emb}")
    # 加载特征
    X, feature_names, rawseq_features = load_features(df, args.biobert_emb, args)
    X_rawseq = rawseq_features

    if args.task == 'regression':
        y = df['activity'].values
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mses, maes, r2s = [], [], []
        y_true_all, y_pred_all = [], []
        best_model = None
        best_r2 = -np.inf
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_rawseq)):
            X_train, X_test = X_rawseq[train_idx], X_rawseq[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model = build_rawseq_cnn_regression(rawseq_maxlen=args.rawseq_maxlen)
            model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=0)
            y_pred = model.predict(X_test).flatten()
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mses.append(mse)
            maes.append(mae)
            r2s.append(r2)
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
            print(f"[回归] Fold {fold+1}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        print(f"[回归] Mean MSE: {np.mean(mses):.4f}, MAE: {np.mean(maes):.4f}, R2: {np.mean(r2s):.4f}")
        # 保存最优模型
        best_model.save(os.path.join(args.out_dir, 'best_RawSeq_CNN_regression.h5'))
        print(f"[回归] Best RawSeq_CNN regression model saved to: {os.path.join(args.out_dir, 'best_RawSeq_CNN_regression.h5')}")
        # 可视化
        plt.figure(figsize=(6,6))
        plt.scatter(y_true_all, y_pred_all, alpha=0.5)
        plt.xlabel('True Activity')
        plt.ylabel('Predicted Activity')
        plt.title('RawSeq_CNN Regression: True vs Predicted')
        plt.plot([min(y_true_all), max(y_true_all)], [min(y_true_all), max(y_true_all)], 'r--')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'rawseq_cnn_regression_scatter.png'))
        plt.close()
    else:
        # 分类任务
        y = (df['activity'] >= args.classify_threshold).astype(int).values
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accs, aucs, f1s = [], [], []
        y_true_all, y_pred_all, y_prob_all = [], [], []
        best_model = None
        best_auc = -np.inf
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, precision_recall_curve
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_rawseq)):
            X_train, X_test = X_rawseq[train_idx], X_rawseq[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model = build_rawseq_cnn_classification(rawseq_maxlen=args.rawseq_maxlen)
            model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=0)
            y_prob = model.predict(X_test).flatten()
            y_pred = (y_prob > 0.5).astype(int)
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            f1 = f1_score(y_test, y_pred)
            accs.append(acc)
            aucs.append(auc)
            f1s.append(f1)
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            y_prob_all.extend(y_prob)
            if auc > best_auc:
                best_auc = auc
                best_model = model
            print(f"[分类] Fold {fold+1}: ACC={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}")
        print(f"[分类] Mean ACC: {np.mean(accs):.4f}, AUC: {np.mean(aucs):.4f}, F1: {np.mean(f1s):.4f}")
        # 保存最优模型
        best_model.save(os.path.join(args.out_dir, 'best_RawSeq_CNN_classification.h5'))
        print(f"[分类] Best RawSeq_CNN classification model saved to: {os.path.join(args.out_dir, 'best_RawSeq_CNN_classification.h5')}")
        # 可视化
        plt.figure(figsize=(6,6))
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true_all, y_pred_all)
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("RawSeq_CNN Confusion Matrix")
        plt.savefig(os.path.join(args.out_dir, 'rawseq_cnn_classification_confusion_matrix.png'))
        plt.close()
        from sklearn.metrics import roc_curve, precision_recall_curve
        fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true_all, y_prob_all):.3f}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("RawSeq_CNN ROC Curve")
        plt.legend()
        plt.savefig(os.path.join(args.out_dir, 'rawseq_cnn_classification_roc.png'))
        plt.close()
        precision, recall, _ = precision_recall_curve(y_true_all, y_prob_all)
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("RawSeq_CNN PR Curve")
        plt.savefig(os.path.join(args.out_dir, 'rawseq_cnn_classification_pr.png'))
        plt.close()

if __name__ == '__main__':
    main()
