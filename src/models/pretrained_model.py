import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow import keras
import torch
from transformers import AutoTokenizer, AutoModel, BertModel, BertConfig
from sklearn.preprocessing import StandardScaler
from src.models.Attention import Attention_layer
import joblib

logger = logging.getLogger(__name__)

class PretrainedModel:
    """预训练模型类，用于加载和微调预训练的抗菌肽模型"""
    
    def __init__(self, model_type='att', pretrained_path=None, feature_dim=768, 
                 finetune_layers=1, dropout_rate=0.3, hidden_dim=64, y_min=0, y_max=1, physchem_dim=0,
                 fusion_mode='concat', physchem_mlp_hidden_dim=32, physchem_mlp_dropout=0.1):
        """
        初始化预训练模型
        
        参数:
        model_type (str): 预训练模型类型，可选值: 'att', 'lstm', 'biobert'
        pretrained_path (str): 预训练模型路径，如果为None则使用默认路径
        feature_dim (int): 特征维度
        finetune_layers (int): 微调的层数，默认微调最后一层
        dropout_rate (float): Dropout比率
        hidden_dim (int): 隐藏层维度
        y_min (float): 标签最小值，用于调整输出范围
        y_max (float): 标签最大值，用于调整输出范围
        physchem_dim (int): 理化特征维度
        fusion_mode (str): 融合方式（'concat'或'mlp_concat'）
        physchem_mlp_hidden_dim (int): 理化特征MLP隐藏层维度
        physchem_mlp_dropout (float): 理化特征MLP Dropout
        """
        self.model_type = model_type
        self.pretrained_path = pretrained_path if pretrained_path else os.path.join("models", "pretrained")
        self.feature_dim = feature_dim
        self.finetune_layers = finetune_layers
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.y_min = y_min
        self.y_max = y_max
        self.physchem_dim = physchem_dim
        self.fusion_mode = fusion_mode
        self.physchem_mlp_hidden_dim = physchem_mlp_hidden_dim
        self.physchem_mlp_dropout = physchem_mlp_dropout
        self.model = None
        self.tokenizer = None  # 用于BioBERT
        self.is_trained = False
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 加载预训练模型
        self._load_pretrained_model()

    def _build_att_model(self):
        """构建与预训练 att.h5 完全一致的模型结构"""
        seq_input = keras.Input(shape=(300,), dtype='int32', name='seq_input')
        x = keras.layers.Embedding(input_dim=21, output_dim=128, input_length=300)(seq_input)
        x = keras.layers.Conv1D(filters=64, kernel_size=16, activation='relu')(x)
        x = keras.layers.MaxPooling1D(pool_size=5)(x)
        x = keras.layers.Dropout(self.dropout_rate)(x)
        x = Attention_layer()(x)
        out = keras.layers.Dense(1)(x)
        return keras.Model(seq_input, out)

    def _build_lstm_model(self):
        """构建与预训练 lstm.h5 完全一致的模型结构"""
        seq_input = keras.Input(shape=(300,), dtype='int32', name='seq_input')
        x = keras.layers.Embedding(input_dim=21, output_dim=128, input_length=300)(seq_input)
        x = keras.layers.Conv1D(filters=64, kernel_size=16, activation='relu')(x)
        x = keras.layers.MaxPooling1D(pool_size=5)(x)
        x = keras.layers.Dropout(self.dropout_rate)(x)
        x = keras.layers.LSTM(100)(x)
        out = keras.layers.Dense(1)(x)
        return keras.Model(seq_input, out)
    
    def _load_pretrained_model(self):
        """加载预训练模型"""
        try:
            if self.model_type.lower() in ['att', 'lstm']:
                model_name = self.model_type.lower()
                model_path = self.pretrained_path
                logger.info(f"加载预训练 {model_name} 模型: {model_path}")

                self.model = (self._build_att_model() if self.model_type.lower()=='att' else self._build_lstm_model())
                self.model.load_weights(model_path, by_name=True)
                # 设置需要微调的层
                total_layers = len(self.model.layers)
                trainable_layers = min(self.finetune_layers, total_layers)
            
                # 默认冻结所有层
                for layer in self.model.layers:
                    layer.trainable = False
            
                # 只解冻最后几层用于微调
                for layer in self.model.layers[-trainable_layers:]:
                    layer.trainable = True
            
                logger.info(f"已加载预训练{model_name}模型，共{total_layers}层，将微调最后{trainable_layers}层")
            
                # 重新编译模型
                self.model.compile(
                    optimizer=keras.optimizers.Adam(),
                    loss='mse',
                    metrics=['mae']
                )

            elif self.model_type.lower() == 'biobert':
                # 加载BioBERT
                model_path = os.path.join(self.pretrained_path, 'biobert')
                logger.info(f"加载预训练BioBERT模型: {model_path}")
            
                # 检查模型目录是否存在，如果不存在则从Hugging Face下载
                if not os.path.exists(model_path):
                    logger.info("本地未找到BioBERT模型，将从Hugging Face下载...")
                    self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
                    self.bert_model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
                
                    # 保存到本地
                    os.makedirs(model_path, exist_ok=True)
                    self.tokenizer.save_pretrained(model_path)
                    self.bert_model.save_pretrained(model_path)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    self.bert_model = AutoModel.from_pretrained(model_path)
            
                # 冻结大部分层
                for param in self.bert_model.parameters():
                    param.requires_grad = False
            
                # 只微调最后几层
                for i, layer in enumerate(self.bert_model.encoder.layer):
                    if i >= len(self.bert_model.encoder.layer) - self.finetune_layers:
                        for param in layer.parameters():
                            param.requires_grad = True
            
                # 获取理化特征维度
                n_physchem = self.physchem_dim

                # 创建支持理化特征拼接/MLP融合的回归头
                class BertRegressionModel(torch.nn.Module):
                    def __init__(self, bert_model, hidden_dim, dropout_rate, n_physchem, fusion_mode='concat', physchem_mlp_hidden_dim=32, physchem_mlp_dropout=0.1):
                        super().__init__()
                        self.bert = bert_model
                        self.dropout = torch.nn.Dropout(dropout_rate)
                        self.fusion_mode = fusion_mode
                        self.n_physchem = n_physchem
                        if fusion_mode == 'mlp_concat' and n_physchem > 0:
                            self.physchem_mlp = torch.nn.Sequential(
                                torch.nn.Linear(n_physchem, physchem_mlp_hidden_dim),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(physchem_mlp_dropout),
                                torch.nn.Linear(physchem_mlp_hidden_dim, physchem_mlp_hidden_dim),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(physchem_mlp_dropout)
                            )
                            fusion_input_dim = self.bert.config.hidden_size + physchem_mlp_hidden_dim
                        else:
                            self.physchem_mlp = None
                            fusion_input_dim = self.bert.config.hidden_size + n_physchem
                        self.fc1 = torch.nn.Linear(fusion_input_dim, hidden_dim)
                        self.fc2 = torch.nn.Linear(hidden_dim, 1)
                    def forward(self, input_ids, attention_mask, physchem_features=None):
                        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                        cls_output = outputs.last_hidden_state[:, 0, :]
                        if physchem_features is not None:
                            if not isinstance(physchem_features, torch.Tensor):
                                physchem_features = torch.tensor(physchem_features, dtype=torch.float32, device=cls_output.device)
                            if physchem_features.device != cls_output.device:
                                physchem_features = physchem_features.to(cls_output.device)
                            if len(physchem_features.shape) == 1:
                                physchem_features = physchem_features.unsqueeze(0)
                            if self.fusion_mode == 'mlp_concat' and self.physchem_mlp is not None:
                                physchem_out = self.physchem_mlp(physchem_features)
                                x = torch.cat([cls_output, physchem_out], dim=1)
                            else:
                                x = torch.cat([cls_output, physchem_features], dim=1)
                        else:
                            x = cls_output
                        x = self.dropout(x)
                        x = torch.nn.functional.relu(self.fc1(x))
                        x = self.dropout(x)
                        x = self.fc2(x)
                        return x
                # 创建完整模型
                self.model = BertRegressionModel(self.bert_model, self.hidden_dim, self.dropout_rate, n_physchem, self.fusion_mode, self.physchem_mlp_hidden_dim, self.physchem_mlp_dropout)
                self.model.to(self.device)            
                logger.info(f"{self.model_type} 预训练模型加载成功 (支持理化特征融合: {self.fusion_mode})")

            elif self.model_type.lower() == 'esm3':
                from src.esm3_patch import ESM3LocalLoader
                logger.info("加载ESM3模型进行直接微调...")
                model_dir = self.pretrained_path
                self.esm3_model, self.esm3_dtype = ESM3LocalLoader.load_esm3_model_local(
                    model_dir=model_dir, 
                    device=self.device
                )
                logger.info("ESM3模型加载成功")
                logger.info(f"ESM3模型类型: {type(self.esm3_model)}")
                class ESM3SimpleRegressionHead(torch.nn.Module):
                    def __init__(self, esm3_model, hidden_dim, dropout_rate):
                        super().__init__()
                        self.esm3 = esm3_model
                        self.dropout = torch.nn.Dropout(dropout_rate)
                        self.emb_dim = 1536
                        self.hidden_dim = hidden_dim
                        self.fc1 = torch.nn.Linear(self.emb_dim, hidden_dim)
                        self.fc2 = torch.nn.Linear(hidden_dim, 1)
                
                    def forward(self, sequences):
                        from src.esm3_patch import ESM3LocalLoader
                        device = next(self.parameters()).device
                        if isinstance(sequences, str):
                            sequences = [sequences]
                        embeds = []
                        for seq in sequences:
                            emb_dict = ESM3LocalLoader.extract_embeddings(self.esm3, seq, device=device)
                            embeds.append(emb_dict['mean'])
                        embeddings = torch.tensor(embeds, dtype=torch.float32, device=device)
                        # 动态创建线性层，适应实际的嵌入维度
                        if not hasattr(self, 'emb_dim_initialized'):
                            self.emb_dim = embeddings.shape[1]
                            self.fc1 = torch.nn.Linear(self.emb_dim, self.hidden_dim).to(device)
                            self.fc2 = torch.nn.Linear(self.hidden_dim, 1).to(device)
                            self.emb_dim_initialized = True         
                        # 回归头处理
                        x = self.dropout(embeddings)
                        x = torch.nn.functional.relu(self.fc1(x))
                        x = self.dropout(x)
                        x = self.fc2(x)
                        return x
                
                # 创建完整模型
                self.model = ESM3SimpleRegressionHead(
                    self.esm3_model, 
                    self.hidden_dim, 
                    self.dropout_rate
                )
                self.model.to(self.device)
    
                logger.info(f"ESM3迁移学习模型创建成功")
    
                for param in self.esm3_model.parameters():
                    param.requires_grad = False
                for param in self.model.fc1.parameters():
                    param.requires_grad = True
                for param in self.model.fc2.parameters():
                    param.requires_grad = True
                
        except Exception as e:
            logger.error(f"加载预训练模型失败: {e}")
            raise RuntimeError(f"无法加载预训练模型: {e}")

    def preprocess_features(self, X, sequences=None):
        """
        预处理特征，使其适合预训练模型的输入格式
        
        参数:
        X: 特征数据
        sequences: 原始氨基酸序列（用于BioBERT）
        """
        if sequences is None:
            raise ValueError(f"{self.model_type} 模型现在需要提供原始序列数据")
        if self.model_type.lower() in ['att', 'lstm']:
            # 处理序列数据 - 将氨基酸序列编码为数字
            aa_dict = {aa: i+1 for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
            encoded_seqs = [[aa_dict.get(aa, 0) for aa in seq] for seq in sequences]
            padded_seqs = keras.preprocessing.sequence.pad_sequences(
                encoded_seqs, maxlen=300, padding='pre'  # 300需与你模型输入shape一致
            )
            # 保证shape与模型一致
            if hasattr(self.model, "input_shape") and len(self.model.input_shape) == 3 and self.model.input_shape[-1] == 1:
                padded_seqs = np.expand_dims(padded_seqs, -1)
            return padded_seqs

        elif self.model_type.lower() == 'biobert':
            # BioBERT需要将序列转换为token IDs
            encoded_inputs = self.tokenizer(
                sequences,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
               
            return {
                'input_ids': encoded_inputs['input_ids'],
                'attention_mask': encoded_inputs['attention_mask']
            }
        
        elif self.model_type.lower() == 'esm3':
            return sequences

    def train(self, X, y, sequences=None, validation_data=None, epochs=10, batch_size=16, 
              learning_rate=0.001, callbacks=None, physchem_features=None, **kwargs):
        """
        微调预训练模型
        
        参数:
        X: 特征数据
        y: 标签数据
        sequences: 原始序列（用于BioBERT）
        validation_data: 验证数据
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        callbacks: 回调函数列表
        """
        if sequences is None or len(sequences)==0 or y is None or len(y)==0:
            logger.error("训练数据为空，无法进行训练")
            raise ValueError("训练数据不能为空")
        
        logger.info(f"开始微调预训练 {self.model_type} 模型，样本数: {len(sequences)}")
        
        # 预处理特征
        processed_X = self.preprocess_features(None, sequences)
        
        # 不同模型类型的训练逻辑
        if self.model_type.lower() in ['att', 'lstm']:
            # 设置TensorFlow/Keras模型的学习率
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            # 默认回调函数
            if callbacks is None:
                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        min_lr=1e-6
                    )
                ]
            
            # 处理验证数据
            val_data = None
            if validation_data:
                val_X, val_y = validation_data
                val_X = self.preprocess_features(val_X, sequences)
                val_data = (val_X, val_y)
            
            # 微调模型
            history = self.model.fit(
                processed_X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2 if val_data is None else 0,
                validation_data=val_data,
                callbacks=callbacks,
                verbose=1
            )
            
        elif self.model_type.lower() == 'esm3':
            # PyTorch训练逻辑
            self.model.train()
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                        lr=learning_rate)
            criterion = torch.nn.MSELoss()
    
            # 创建数据集
            from torch.utils.data import Dataset, DataLoader
    
            class SequenceDataset(Dataset):
                def __init__(self, sequences, labels):
                    self.sequences = sequences
                    self.labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
            
                def __len__(self):
                    return len(self.sequences)
            
                def __getitem__(self, idx):
                    return self.sequences[idx], self.labels[idx]
    
            # 创建数据加载器
            dataset = SequenceDataset(sequences, y)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            history = {"loss": []}
    
            # 训练循环
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_sequences, batch_y in train_loader:
                    batch_y = batch_y.to(self.device)
            
                    # 前向传播
                    outputs = self.model(batch_sequences)
                    loss = criterion(outputs, batch_y)
            
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
                    epoch_loss += loss.item()
        
                avg_loss = epoch_loss / len(train_loader)
                history["loss"].append(avg_loss)
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        elif self.model_type.lower() == 'biobert':
            if torch.cuda.is_available():
                logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB总量，"
                            f"{torch.cuda.memory_allocated() / 1e9:.2f} GB已分配")
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = torch.nn.MSELoss()
            input_ids = processed_X['input_ids'].to(self.device)
            attention_mask = processed_X['attention_mask'].to(self.device)
            # 理化特征处理
            if physchem_features is not None:
                physchem_tensor = torch.tensor(physchem_features, dtype=torch.float32, device=self.device)
            else:
                physchem_tensor = None
            y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device)
            from torch.utils.data import TensorDataset, DataLoader
            if physchem_tensor is not None:
                dataset = TensorDataset(input_ids, attention_mask, physchem_tensor, y_tensor)
            else:
                dataset = TensorDataset(input_ids, attention_mask, y_tensor)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for epoch in range(epochs):
                epoch_loss = 0
                for batch in train_loader:
                    if physchem_tensor is not None:
                        batch_input_ids, batch_attention_mask, batch_physchem, batch_y = batch
                    else:
                        batch_input_ids, batch_attention_mask, batch_y = batch
                        batch_physchem = None
                    outputs = self.model(batch_input_ids, batch_attention_mask, batch_physchem)
                    loss = criterion(outputs, batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                avg_loss = epoch_loss / len(train_loader)
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            history = {"loss": []}
        
        self.is_trained = True
        return history

    def predict(self, X, sequences=None, physchem_features=None):
        """
        使用微调后的模型进行预测
        
        参数:
        X: 特征数据
        sequences: 原始序列（用于BioBERT）
        physchem_features: 理化特征（可选）
        """
        if not self.is_trained and self.model_type != 'biobert':  # BioBERT可以直接用预训练模型
            logger.warning("模型尚未微调，可能会导致预测结果不准确")
        
        # 预处理特征
        processed_X = self.preprocess_features(None, sequences)
        # 兼容理化特征（如有需要，可在此处拼接或处理）
        # 目前保留接口，后续可扩展
        
        # 不同模型类型的预测逻辑
        if self.model_type.lower() in ['att', 'lstm']:
            predictions = self.model.predict(processed_X).flatten()
            return predictions
        
        elif self.model_type.lower() == 'biobert':
            self.model.eval()
            processed_X = self.preprocess_features(None, sequences)
            input_ids = processed_X['input_ids'].to(self.device)
            attention_mask = processed_X['attention_mask'].to(self.device)
            if physchem_features is not None:
                physchem_tensor = torch.tensor(physchem_features, dtype=torch.float32, device=self.device)
            else:
                physchem_tensor = None
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask, physchem_tensor)
                predictions = outputs.cpu().numpy().flatten()
            return predictions
        
        elif self.model_type.lower() == 'esm3':
            self.model.eval()
            batch_size = 8
            num_sequences = len(sequences)
            predictions = []
    
            with torch.no_grad():
                 # 分批处理，避免内存不足
                for i in range(0, num_sequences, batch_size):
                    batch_seqs = sequences[i:i+batch_size]
                    outputs = self.model(batch_seqs)
                    batch_preds = outputs.cpu().numpy().flatten()
                    predictions.extend(batch_preds)
    
            return predictions
    
    def evaluate(self, X, y, sequences=None, physchem_features=None):
        """
        评估模型性能
        
        参数:
        X: 特征数据
        y: 标签数据
        sequences: 原始序列（用于BioBERT）
        physchem_features: 理化特征（可选）
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # 获取预测结果
        y_pred = self.predict(X, sequences, physchem_features=physchem_features)
        
        # 计算评估指标
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    def save(self, model_dir=None):
        """保存微调后的模型"""
        if model_dir is None:
            model_dir = os.path.join("models", "finetuned", self.model_type)
        
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存标准化器
        joblib.dump(self.scaler, os.path.join(model_dir, f'{self.model_type}_scaler.pkl'))
        
        # 保存模型超参数
        with open(os.path.join(model_dir, f'{self.model_type}_config.txt'), 'w') as f:
            f.write(f"Model type: {self.model_type}\n")
            f.write(f"Feature dimension: {self.feature_dim}\n")
            f.write(f"Finetune layers: {self.finetune_layers}\n")
            f.write(f"Hidden dimension: {self.hidden_dim}\n")
            f.write(f"Dropout rate: {self.dropout_rate}\n")
            f.write(f"y_min: {self.y_min}\n")
            f.write(f"y_max: {self.y_max}\n")
        
        # 根据模型类型保存
        if self.model_type.lower() in ['att', 'lstm']:
            model_path = os.path.join(model_dir, f"{self.model_type}_finetuned.h5")
            if self.model_type.lower() == 'att':
                self.model.save(model_path, save_format='h5')
            else:
                self.model.save(model_path)
            
        elif self.model_type.lower() == 'biobert':
            model_path = os.path.join(model_dir, f"{self.model_type}_finetuned")
            os.makedirs(model_path, exist_ok=True)
            
            # 保存模型状态和tokenizer
            torch.save(self.model.state_dict(), os.path.join(model_path, 'pytorch_model.bin'))
            self.tokenizer.save_pretrained(model_path)
            
            # 保存一些配置信息
            config = {
                "hidden_dim": self.hidden_dim,
                "dropout_rate": self.dropout_rate
            }
            import json
            with open(os.path.join(model_path, 'finetuning_config.json'), 'w') as f:
                json.dump(config, f, indent=2)
        
        logger.info(f"{self.model_type} 模型已保存到 {model_dir}")
    
    def load(self, model_dir=None):
        """加载微调后的模型"""
        if model_dir is None:
            model_dir = os.path.join("models", "finetuned", self.model_type)
        
        # 加载标准化器
        scaler_path = os.path.join(model_dir, f'{self.model_type}_scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        # 根据模型类型加载
        if self.model_type.lower() in ['att', 'lstm']:
            model_path = os.path.join(model_dir, f"{self.model_type}_finetuned.h5")
            if os.path.exists(model_path):
                if self.model_type.lower() == 'att':
                    self.model = keras.models.load_model(
                        model_path,
                        custom_objects={'Attention_layer': Attention_layer}
                    )
                else:
                    self.model = keras.models.load_model(model_path)
                self.is_trained = True
                logger.info(f"已加载微调后的 {self.model_type} 模型: {model_path}")
            else:
                raise FileNotFoundError(f"找不到微调模型文件: {model_path}")
                
        elif self.model_type.lower() == 'biobert':
            model_path = os.path.join(model_dir, f"{self.model_type}_finetuned")
            
            # 加载配置
            import json
            with open(os.path.join(model_path, 'finetuning_config.json'), 'r') as f:
                config = json.load(f)
            
            self.hidden_dim = config["hidden_dim"]
            self.dropout_rate = config["dropout_rate"]
            
            # 重新创建模型架构
            self._load_pretrained_model()
            
            # 加载保存的权重
            self.model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin')))
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.is_trained = True
            logger.info(f"已加载微调后的 {self.model_type} 模型: {model_path}")