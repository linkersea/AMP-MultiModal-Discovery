import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import optuna
from src.features.feature_extractor import FeatureExtractor

def binarize_labels(df, threshold=0.8): #活性阈值
    df['label'] = (df['activity'] >= threshold).astype(int)
    return df

def extract_physchem_features(input_csv, output_csv):
    fe = FeatureExtractor()
    fe.batch_extract_and_save(input_csv, seq_col='sequence', save_path=output_csv)

def extract_biobert_embeddings(sequences, model_dir, output_path):
    from transformers import AutoTokenizer, AutoModel
    import torch
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for seq in tqdm(sequences, desc="提取BioBERT嵌入"):
            inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_emb)
    embeddings = np.array(embeddings)
    np.save(output_path, embeddings)
    return embeddings

def extract_rawseq_features(seqs, max_len=20):
    # 氨基酸字典，A=1, C=2, ..., Y=20, 0为padding
    aa_dict = {aa: i+1 for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    seq_ids = []
    for seq in seqs:
        ids = [aa_dict.get(aa, 0) for aa in seq.upper()[:max_len]]
        ids = ids + [0]*(max_len - len(ids)) if len(ids) < max_len else ids[:max_len]
        seq_ids.append(ids)
    return np.array(seq_ids, dtype=np.int32)

def get_classifiers():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "SVM": SVC(probability=True),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier()
    }

def get_dl_models(input_dim, num_classes=2, use_rawseq=False, rawseq_maxlen=20, vocab_size=21, embedding_dim=32):
    from tensorflow import keras
    models = {}
    # MLP
    if not use_rawseq:
        mlp = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', keras.metrics.AUC()])
        models['MLP'] = mlp
    # 1D CNN/LSTM/GRU for rawseq
    if use_rawseq:
        # Embedding+CNN
        cnn = keras.Sequential([
            keras.layers.Input(shape=(rawseq_maxlen,)),
            keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=rawseq_maxlen),
            keras.layers.Conv1D(64, 3, activation='relu'),
            keras.layers.MaxPooling1D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', keras.metrics.AUC()])
        models['RawSeq_CNN'] = cnn
        # Embedding+LSTM
        lstm = keras.Sequential([
            keras.layers.Input(shape=(rawseq_maxlen,)),
            keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=rawseq_maxlen),
            keras.layers.LSTM(64, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', keras.metrics.AUC()])
        models['RawSeq_LSTM'] = lstm
        # Embedding+GRU
        gru = keras.Sequential([
            keras.layers.Input(shape=(rawseq_maxlen,)),
            keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=rawseq_maxlen),
            keras.layers.GRU(64, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', keras.metrics.AUC()])
        models['RawSeq_GRU'] = gru
    # 1D CNN/LSTM/GRU for feature vector
    if not use_rawseq:
        cnn = keras.Sequential([
            keras.layers.Input(shape=(input_dim, 1)),
            keras.layers.Conv1D(64, 3, activation='relu'),
            keras.layers.MaxPooling1D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', keras.metrics.AUC()])
        models['1D_CNN'] = cnn
        lstm = keras.Sequential([
            keras.layers.Input(shape=(input_dim, 1)),
            keras.layers.LSTM(64, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', keras.metrics.AUC()])
        models['LSTM'] = lstm
        gru = keras.Sequential([
            keras.layers.Input(shape=(input_dim, 1)),
            keras.layers.GRU(64, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', keras.metrics.AUC()])
        models['GRU'] = gru
    return models

def load_features(args, df, biobert_emb_path=None):
    features = []
    feature_names = []
    fe = FeatureExtractor()
    # 解析特征组合
    feature_set = set([f.strip().lower() for f in args.features.split('+')])
    
    if 'physchem' in feature_set:
        physchem_cols = [col for col in df.columns if col not in ['sequence', 'activity', 'label']]
        features.append(df[physchem_cols].values)
        feature_names += physchem_cols
    
    if 'seqeng' in feature_set:
        # 统一使用 FeatureExtractor 的方法进行序列工程特征提取
        # 基础序列特征
        seq_features = np.array([list(fe.extract_features(seq).values()) for seq in df['sequence']])
        features.append(seq_features)
        feature_names += list(fe.extract_features(df['sequence'].iloc[0]).keys())
        
        # n-gram 特征
        ngram_features, ngram_names = fe.extract_ngram_features(df['sequence'], n=args.ngram_n)
        features.append(ngram_features)
        feature_names += [f"ngram_{ng}" for ng in ngram_names]
        
        # window AAC 特征
        window_aac_features = fe.extract_window_aac(df['sequence'], window=args.window_size)
        features.append(window_aac_features)
        feature_names += [f"winAAC_{i+1}" for i in range(window_aac_features.shape[1])]
        
        # terminal 特征
        terminal_features = fe.extract_terminal_features(df['sequence'], n=args.terminal_n)
        features.append(terminal_features)
        feature_names += [f"Nterm_{aa}" for aa in fe.AA_LIST] + [f"Cterm_{aa}" for aa in fe.AA_LIST]
        
        # PSSM 特征
        if 'pssm' in feature_set or args.pssm_dir:
            pssm_features = fe.extract_pssm_features(df['sequence'], pssm_dir=args.pssm_dir)
            features.append(pssm_features)
            feature_names += [f"PSSM_{aa}" for aa in fe.AA_LIST]
    if 'biobert' in feature_set and biobert_emb_path:
        biobert_emb = np.load(biobert_emb_path)
        features.append(biobert_emb)
        feature_names += [f"biobert_{i}" for i in range(biobert_emb.shape[1])]
    # 原始序列token id特征仅深度学习模型用，默认启用，除非--no_rawseq
    use_rawseq = not getattr(args, 'no_rawseq', False)
    if use_rawseq and not getattr(args, 'no_dl', False):
        rawseq_features = fe.extract_rawseq_features(df['sequence'], max_len=args.rawseq_maxlen)
        features.append(rawseq_features)
        feature_names += [f"pos_{i+1}" for i in range(rawseq_features.shape[1])]
    X = np.concatenate(features, axis=1) if len(features) > 1 else features[0]
    return X, feature_names

def plot_metrics(y_true, y_pred, y_prob, out_dir, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(os.path.join(out_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_prob):.3f}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{model_name} ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{model_name}_roc.png"))
    plt.close()
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} PR Curve")
    plt.savefig(os.path.join(out_dir, f"{model_name}_pr.png"))
    plt.close()

def feature_selection(X, y, method='kbest', k=16):
    if method == 'kbest':
        selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
        X_new = selector.fit_transform(X, y)
        return X_new, selector
    elif method == 'mutual_info':
        selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
        X_new = selector.fit_transform(X, y)
        return X_new, selector
    elif method == 'pca':
        pca = PCA(n_components=min(k, X.shape[1]))
        X_new = pca.fit_transform(X)
        return X_new, pca
    else:
        return X, None

def optuna_tune(clf, X, y, model_name, n_trials=30, feature_selection_method='kbest', k_range=(8, 128)):
    def objective(trial):
        params = {}
        # k自动调参
        k = trial.suggest_int('k', k_range[0], min(k_range[1], X.shape[1]))
        X_new, _ = feature_selection(X, y, method=feature_selection_method, k=k)
        if model_name == 'RandomForest':
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
            params['max_depth'] = trial.suggest_int('max_depth', 2, 16)
        elif model_name == 'SVM':
            params['C'] = trial.suggest_float('C', 0.01, 10, log=True)
            params['kernel'] = trial.suggest_categorical('kernel', ['linear', 'rbf'])
        elif model_name == 'XGBoost':
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
            params['max_depth'] = trial.suggest_int('max_depth', 2, 16)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3)
        elif model_name == 'LightGBM':
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
            params['max_depth'] = trial.suggest_int('max_depth', 2, 16)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3)
        elif model_name == 'LogisticRegression':
            params['C'] = trial.suggest_float('C', 0.01, 10, log=True)
        clf.set_params(**params)
        scores = cross_val_score(clf, X_new, y, cv=3, scoring='roc_auc')
        return np.mean(scores)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    # 用最优k重新特征选择
    best_k = study.best_params['k'] if 'k' in study.best_params else min(16, X.shape[1])
    X_new, selector = feature_selection(X, y, method=feature_selection_method, k=best_k)
    clf.set_params(**{k: v for k, v in study.best_params.items() if k != 'k'})
    return clf, study, X_new, selector

def plot_feature_importance(clf, feature_names, out_dir, model_name):
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        idx = np.argsort(importances)[::-1][:20]
        plt.figure(figsize=(8, 6))
        plt.barh(np.array(feature_names)[idx][::-1], importances[idx][::-1])
        plt.title(f"{model_name} Top 20 Feature Importances")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{model_name}_feature_importance.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_csv', type=str, default='data/raw/120dataset.csv')
    parser.add_argument('--physchem_csv', type=str, default='data/processed/120dataset_physchem.csv')
    parser.add_argument('--biobert_dir', type=str, default='model/biobert/biobert')
    parser.add_argument('--biobert_emb', type=str, default='data/processed/biobert_emb.npy')
    parser.add_argument('--features', type=str, default='physchem+seqeng',
                        help='特征组合，支持physchem, seqeng, physchem+seqeng, biobert, pssm等')
    parser.add_argument('--no_dl', action='store_true', help='不运行深度学习模型')
    parser.add_argument('--no_rawseq', action='store_true', help='深度学习时不使用原始序列token id')
    parser.add_argument('--out_dir', type=str, default='results/classification')
    parser.add_argument('--epochs', type=int, default=30, help='深度学习模型训练轮数')
    parser.add_argument('--feature_selection', type=str, choices=['none', 'kbest', 'mutual_info', 'pca'], default='none')
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--n_trials', type=int, default=30)
    parser.add_argument('--rawseq_maxlen', type=int, default=20, help='原始序列最大长度')
    parser.add_argument('--ngram_n', type=int, default=3, help='n-gram的n')
    parser.add_argument('--window_size', type=int, default=5, help='滑动窗口大小')
    parser.add_argument('--terminal_n', type=int, default=3, help='N/C端长度')
    parser.add_argument('--pssm_dir', type=str, default=None, help='PSSM文件目录')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.raw_csv)
    df = binarize_labels(df, threshold=0.8)#活性阈值
    df.to_csv(os.path.join(args.out_dir, "dataset_binarized.csv"), index=False)

    if not os.path.exists(args.physchem_csv):
        extract_physchem_features(args.raw_csv, args.physchem_csv)
    df_physchem = pd.read_csv(args.physchem_csv)
    df = df.merge(df_physchem, on=['sequence', 'activity'], how='left')

    if 'biobert' in args.features and not os.path.exists(args.biobert_emb):
        extract_biobert_embeddings(df['sequence'], args.biobert_dir, args.biobert_emb)

    X, feature_names = load_features(args, df, args.biobert_emb if 'biobert' in args.features else None)
    y = df['label'].values

    if args.feature_selection != 'none':
        # 如果自动调参且特征选择不是none，则k由optuna自动调参
        if args.tune:
            pass  # 由optuna_tune内部处理特征选择
        else:
            X, selector = feature_selection(X, y, method=args.feature_selection, k=args.k)
            if hasattr(selector, 'get_support'):
                feature_names = np.array(feature_names)[selector.get_support()].tolist()
            elif hasattr(selector, 'components_'):
                feature_names = [f"PC{i+1}" for i in range(X.shape[1])]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    all_metrics = {}    # 传统ML模型默认运行
    for model_name, clf in get_classifiers().items():
        accs, aucs, f1s = [], [], []
        y_true_all, y_pred_all, y_prob_all = [], [], []
        X_used = X
        feature_names_used = feature_names
        selector = None  # 初始化selector
        study = None     # 初始化study
        if args.tune:
            clf, study, X_used, selector = optuna_tune(clf, X, y, model_name, n_trials=args.n_trials, feature_selection_method=args.feature_selection, k_range=(8, 128))
            joblib.dump(study, os.path.join(args.out_dir, f"{model_name}_optuna_study.pkl"))
            # 更新特征名
            if hasattr(selector, 'get_support'):
                feature_names_used = np.array(feature_names)[selector.get_support()].tolist()
            elif hasattr(selector, 'components_'):
                feature_names_used = [f"PC{i+1}" for i in range(X_used.shape[1])]
        for train_idx, test_idx in skf.split(X_used, y):
            X_train, X_test = X_used[train_idx], X_used[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
            accs.append(accuracy_score(y_test, y_pred))
            aucs.append(roc_auc_score(y_test, y_prob))
            f1s.append(f1_score(y_test, y_pred))
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            y_prob_all.extend(y_prob)        # 计算best_k值
        best_k = None
        if args.tune and study and 'k' in study.best_params:
            best_k = study.best_params['k']
        elif selector and hasattr(selector, 'k'):
            best_k = selector.k
        elif args.feature_selection != 'none' and not args.tune:
            best_k = args.k
            
        results.append({
            "model": model_name,
            "acc": np.mean(accs),
            "auc": np.mean(aucs),
            "f1": np.mean(f1s),
            "best_k": best_k
        })
        all_metrics[model_name] = {
            'y_true': y_true_all,
            'y_pred': y_pred_all,
            'y_prob': y_prob_all
        }
        plot_metrics(y_true_all, y_pred_all, y_prob_all, args.out_dir, model_name)
        plot_feature_importance(clf, feature_names_used, args.out_dir, model_name)
        print(f"{model_name}: ACC={np.mean(accs):.3f}, AUC={np.mean(aucs):.3f}, F1={np.mean(f1s):.3f}")

    # 深度学习模型默认运行，除非--no_dl
    if not args.no_dl:
        from tensorflow import keras
        use_rawseq = not getattr(args, 'no_rawseq', False)
        rawseq_maxlen = getattr(args, 'rawseq_maxlen', 20)
        dl_models = get_dl_models(X.shape[1] if not use_rawseq else rawseq_maxlen, num_classes=2, use_rawseq=use_rawseq, rawseq_maxlen=rawseq_maxlen)
        best_auc = -1
        best_model = None
        best_model_name = None
        for model_name, model in dl_models.items():
            accs, aucs, f1s = [], [], []
            y_true_all, y_pred_all, y_prob_all = [], [], []
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                # Embedding模型输入为(rawseq_maxlen,)
                if use_rawseq and model_name.startswith('RawSeq_'):
                    X_train_dl = extract_rawseq_features(df.iloc[train_idx]['sequence'], max_len=rawseq_maxlen)
                    X_test_dl = extract_rawseq_features(df.iloc[test_idx]['sequence'], max_len=rawseq_maxlen)
                elif not use_rawseq and model_name in ['1D_CNN', 'LSTM', 'GRU']:
                    X_train_dl = X_train.reshape(-1, X.shape[1], 1)
                    X_test_dl = X_test.reshape(-1, X.shape[1], 1)
                else:
                    X_train_dl, X_test_dl = X_train, X_test
                model.fit(X_train_dl, y_train, epochs=args.epochs, batch_size=16, verbose=0)
                y_prob = model.predict(X_test_dl).flatten()
                y_pred = (y_prob > 0.5).astype(int)
                accs.append(accuracy_score(y_test, y_pred))
                aucs.append(roc_auc_score(y_test, y_prob))
                f1s.append(f1_score(y_test, y_pred))
                y_true_all.extend(y_test)
                y_pred_all.extend(y_pred)
                y_prob_all.extend(y_prob)
            # 保存AUC最优模型
            mean_auc = np.mean(aucs)
            if model_name == 'RawSeq_CNN' and mean_auc > best_auc:
                best_auc = mean_auc
                best_model = model
                best_model_name = model_name
            results.append({
                "model": model_name,
                "acc": np.mean(accs),
                "auc": mean_auc,
                "f1": np.mean(f1s),
                "best_k": None  # DL模型不使用特征选择
            })
            all_metrics[model_name] = {
                'y_true': y_true_all,
                'y_pred': y_pred_all,
                'y_prob': y_prob_all
            }
            plot_metrics(y_true_all, y_pred_all, y_prob_all, args.out_dir, model_name)
            print(f"{model_name}: ACC={np.mean(accs):.3f}, AUC={mean_auc:.3f}, F1={np.mean(f1s):.3f}")
        # 自动保存AUC最优RawSeq_CNN模型
        if best_model is not None:
            save_path = os.path.join(args.out_dir, f"best_RawSeq_CNN.h5")
            best_model.save(save_path)
            print(f"Best RawSeq_CNN model saved to: {save_path}")

    pd.DataFrame(results).to_csv(os.path.join(args.out_dir, "model_comparison.csv"), index=False)
    plt.figure(figsize=(8, 5))
    for model_name in all_metrics:
        fpr, tpr, _ = roc_curve(all_metrics[model_name]['y_true'], all_metrics[model_name]['y_prob'])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc_score(all_metrics[model_name]['y_true'], all_metrics[model_name]['y_prob']):.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "roc_comparison.png"))
    plt.close()
    plt.figure(figsize=(8, 5))
    for model_name in all_metrics:
        precision, recall, _ = precision_recall_curve(all_metrics[model_name]['y_true'], all_metrics[model_name]['y_prob'])
        plt.plot(recall, precision, label=model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "pr_comparison.png"))
    plt.close()

if __name__ == "__main__":
    main()
