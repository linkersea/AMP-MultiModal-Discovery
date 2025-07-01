# AMP-MultiModal-Discovery: 抗菌肽多模态智能发现系统

## 项目概述

本项目实现了一个基于多模态深度学习的抗菌肽（AMP）智能发现框架，集成BioBERT嵌入、深度学习分类模型和多种序列生成策略，能够自动化发现具有高抗菌活性的新型多肽序列。

## 核心功能特性

- 🔬 **多特征融合分类模型**: 整合理化特征、序列工程特征、BioBERT嵌入和原始序列CNN
- 🧬 **三种序列生成策略**: 序列变异、理性设计、VAE深度生成
- 🤖 **多模态深度学习筛选**: 使用集成BioBERT预训练嵌入与多肽多模态特征的CNN分类器进行智能评分
- 📊 **端到端自动化**: 从数据预处理到最终候选序列推荐的完整流程
- 🔄 **反馈循环优化**: VAE生成器与多模态深度学习分类器的实时反馈集成

## 技术架构

### 1. 核心分类模型 (PhysChemSeqEngBioBERT+RawSeq CNN)

**特征组合:**
- **理化特征**: 氨基酸组成、二肽组成、分子量、疏水性、等电点等
- **序列工程特征**: N-gram特征、窗口AAC、末端特征
- **BioBERT嵌入**: 利用预训练的生物医学BERT模型捕捉语义信息
- **原始序列CNN**: 直接从序列学习高级表示

**模型架构:**
```
Input: [Raw Sequence (CNN分支)] + [Traditional Features (Dense分支)]
  │                                      │
  ▼                                      ▼
CNN层 (Embedding→Conv1D→MaxPool)    Dense层 (全连接网络)
  │                                      │
  ▼                                      ▼
  Flatten → Concatenate ← Dense Features
            │
            ▼
        Dense Output → Sigmoid → Prediction
```

### 2. 多肽序列生成策略

#### 方法1: 序列变异 (Sequence Variation)
- **原理**: 基于已知高活性序列进行局部探索
- **策略**: 替换、插入、删除等变异操作
- **优势**: 保留已验证序列的核心活性模式

#### 方法2: 理性设计 (Rational Design)
- **原理**: 数据驱动的结构感知设计，结合训练数据统计分析与生物学知识
- **数据洞察**: 
  - 氨基酸活性贡献分析
  - 高活性motif模式识别
  - 位置特异性统计分析
- **结构策略**: 
  - 两亲性螺旋结构 (40%): 膜表面结合和扰动
  - β折叠结构 (35%): 跨膜孔道形成
  - 无规卷曲结构 (25%): 灵活性和适应性
- **设计特点**: 基于统计数据指导的结构生物学设计，兼具科学依据和数据支撑

#### 方法3: VAE生成 (Variational Autoencoder)
- **原理**: 深度学习驱动的序列创新生成
- **架构**: 编码器-解码器结构，潜在空间连续表示
- **优势**: 能发现数据分布之外的新颖序列

### 3. AI反馈优化循环

```
序列生成 → 多模态深度学习分类器预测 → 质量评分 → 筛选决策 → 结果输出
    ↑                                                    ↓
    ←——————————— 反馈调整参数 ←———————————————————————————————
```

## 项目文件结构

```
AMP-MultiModal-Discovery/
├── src/                                    # 源代码目录
│   ├── features/
│   │   └── feature_extractor.py          # 特征提取器
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logger_config.py              # 日志配置
│   ├── advanced_vae_generator.py          # VAE多肽生成器
│   ├── predict_peptide.py                # 多肽活性预测模块
│   ├── rational_design_peptide.py        # 数据驱动的理性设计分析模块
│   ├── peptide_classification_pipeline.py # 分类模型数据结构与模型选择对比管道
│   └── physchem_seqeng_biobert_dl_rawseq_cv_fixed.py #最佳模型五折交叉验证与保存
├── data/                                  # 数据目录
│   ├── raw/
│   │   ├── 120dataset.csv               # 训练数据集
│   │   └── 11pep.csv                    # 测试序列
│   └── processed/                        # 预处理后的数据
├── results/                              # 模型保存目录
│   └── physchem_seqeng_biobert_dl_rawseq/
│       ├── best_physchem_seqeng_biobert_rawseq_classification.h5  # 最佳模型
│       ├── best_classification_scaler.pkl  # 特征标准化器
│       └── window_aac_dim.npy            # 窗口AAC特征维度
├── model/
│   └── biobert/                          # BioBERT预训练模型
├── three_method_discovery.py             # 主发现框架
├── calculate_grafting_density.py         # 接枝密度计算
├── analyze_results.py                   # 结果分析
└── rational_design_analysis_report.txt   # 理性设计分析报告
```

### 特殊依赖
```bash
# BioBERT模型(自动下载)
# 可选: iFeature (用于高级特征提取)
# 需要手动下载并放置在temp/iFeature/目录
```

## 使用指南

### 1. 快速开始 - 运行完整发现流程

```bash
# 基础运行
python three_method_discovery.py

# 自定义参数运行
python three_method_discovery.py \
  --sv_count 300 \
  --rd_count 200 \
  --vae_count 250 \
  --model_path results/physchem_seqeng_biobert_dl_rawseq/best_physchem_seqeng_biobert_rawseq_classification.h5
```

### 2. 单独使用分类器预测

```bash
python src/predict_peptide.py \
  --model_path results/physchem_seqeng_biobert_dl_rawseq/best_physchem_seqeng_biobert_rawseq_classification.h5 \
  --input data/raw/11pep.csv \
  --output pred_result.csv
```

### 3. 训练新的分类模型

```bash
# 注意：模型训练脚本已从公开仓库移除，如需训练请联系作者
# 或参考项目论文中的模型架构自行实现
```

### 4. 单独运行理性设计数据分析

```bash
# 运行数据分析，生成统计报告
python src/rational_design_peptide.py

# 分析结果将保存到 rational_design_analysis_report.txt
# 数据洞察供主框架的结构感知设计使用
```

### 5. 快速演示系统功能

```bash
# 运行演示脚本，包含预测和发现的完整示例
python demo.py

# 演示包含：
# 1. 使用示例序列进行活性预测
# 2. 运行小规模的三方法发现流程
```

## 理性设计数据分析

系统会自动执行全面的数据驱动分析，为结构感知设计提供科学依据：

### 1. 氨基酸活性贡献分析
- 计算每种氨基酸的平均活性贡献
- 统计频率分布和标准差
- 识别高活性氨基酸集合

### 2. Motif模式识别
- 2-4氨基酸motif的活性分析
- 高活性motif的频率统计
- 为序列设计提供功能片段

### 3. 位置特异性分析
- 针对最常见序列长度的位置分析
- 每个位置的最优氨基酸识别
- 理论最优序列构建

### 4. 自动化报告生成
分析完成后会生成 `rational_design_analysis_report.txt`，包含：
- 数据概况和统计信息
- Top 10 高活性氨基酸排序
- Top 10 高活性motif模式
- 理论最优序列信息

## 核心算法原理

### 1. 特征工程

#### 理化特征提取
- **氨基酸组成(AAC)**: 20种氨基酸的频率分布
- **二肽组成(DPC)**: 400种二肽的出现频率
- **生物物理特性**: 分子量、芳香性、不稳定性指数、等电点、疏水性
- **二级结构倾向**: α-螺旋、β-转角、β-折叠比例

#### 序列工程特征
- **N-gram特征**: 捕捉局部序列模式(默认trigram)
- **窗口AAC**: 滑动窗口内的氨基酸组成
- **末端特征**: N端和C端的氨基酸组成

#### BioBERT嵌入
```python
# 序列预处理: "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"
# 转换为: "K W K L F K K I E K V G Q N I R D G I I K A G P A V A V V G Q A T Q I A K"
# BioBERT编码 → 768维向量
```

### 2. CNN分类架构

```python
# 原始序列分支
sequence_input = Input(shape=(max_length,))
embedding = Embedding(vocab_size, 128)(sequence_input)
conv1d = Conv1D(filters=64, kernel_size=3, activation='relu')(embedding)
maxpool = MaxPooling1D(pool_size=2)(conv1d)
flatten = Flatten()(maxpool)

# 传统特征分支
feature_input = Input(shape=(feature_dim,))
dense1 = Dense(256, activation='relu')(feature_input)
dense2 = Dense(128, activation='relu')(dense1)

# 特征融合
concat = Concatenate()([flatten, dense2])
output = Dense(1, activation='sigmoid')(concat)
```

### 3. VAE生成原理

#### 编码器(Encoder)
```
序列 → 嵌入 → LSTM → [μ, σ²] (潜在变量参数)
```

#### 重参数化技巧
```
z = μ + σ * ε, 其中 ε ~ N(0,1)
```

#### 解码器(Decoder)
```
z → LSTM → 输出分布 → 采样序列
```

#### 损失函数
```
L = 重构损失 + β * KL散度损失
重构损失 = CrossEntropy(重构序列, 原序列)
KL损失 = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
```

### 4. 生物学评分规则

```python
def biological_score(sequence):
    score = 0
    
    # 净电荷 (理想范围: +2 到 +6)
    net_charge = count('R','K','H') - count('D','E')
    if 2 <= net_charge <= 6: score += 25
    
    # 疏水性比例 (理想范围: 30%-60%)
    hydrophobic_ratio = count('I','L','M','F','W','Y','V') / length
    if 0.3 <= hydrophobic_ratio <= 0.6: score += 20
    
    # 芳香族氨基酸存在
    aromatic = count('F','W','Y')
    if aromatic >= 1: score += 15
    
    # 长度优化 (理想范围: 10-14氨基酸)
    if 10 <= length <= 14: score += 15
    
    # 关键氨基酸奖励
    if 'R' in sequence: score += 10  # 精氨酸
    if 'W' in sequence: score += 8   # 色氨酸
    
    return score
```

## 输出结果说明

### 1. 目录结构
运行完成后会生成 `results_three_methods_YYYYMMDD_HHMMSS/` 目录，包含：

```
results_three_methods_20241226_143052/
├── sequence_variation/                    # 序列变异结果
│   ├── sequence_variation_candidates_20241226_143052.csv
│   └── sequence_variation_log_20241226_143052.json
├── rational_design/                       # 理性设计结果
│   ├── rational_design_candidates_20241226_143052.csv
│   └── rational_design_log_20241226_143052.json
├── vae_generation/                        # VAE生成结果
│   ├── vae_generation_candidates_20241226_143052.csv
│   └── vae_generation_log_20241226_143052.json
├── final_predicted_candidates_20241226_143052.csv  # 最终排序结果
└── peptide_discovery_report_20241226_143052.md     # 综合报告
```

### 2. CSV结果文件说明

#### 最终候选文件字段
| 字段名 | 说明 | 示例值 |
|--------|------|--------|
| sequence | 多肽序列 | "KWKLFKKIEKVGQ" |
| method | 生成方法 | "sequence_variation" |
| predicted_activity | AI预测活性概率 | 0.8542 |
| biological_score | 生物学规则评分 | 75 |
| length | 序列长度 | 13 |
| generation_strategy | 生成策略 | "local_search" |

### 3. 评分解读

#### AI预测活性概率
- **> 0.8**: 高度推荐合成测试
- **0.6-0.8**: 中等推荐
- **0.4-0.6**: 需要进一步验证
- **< 0.4**: 不推荐

#### 生物学规则评分
- **≥ 70**: 理论上具有良好抗菌潜力
- **50-69**: 中等潜力，可考虑结构优化
- **< 50**: 活性较低

## 高级功能

### 1. 自定义反馈模型

```python
# 使用自己的分类器
from three_method_discovery import ThreeMethodPeptideDiscovery

discovery = ThreeMethodPeptideDiscovery(
    model_path="path/to/your/model.h5",
    scaler_path="path/to/your/scaler.pkl"
)
```

### 2. 批量预测新序列

```python
from src.predict_peptide import PhysChemSeqEngBioBERTPredictor

predictor = PhysChemSeqEngBioBERTPredictor(
    "results/physchem_seqeng_biobert_dl_rawseq/best_physchem_seqeng_biobert_rawseq_classification.h5"
)

sequences = ["KWKLFKKIEKVGQ", "RRWWRF"]
y_prob, y_pred = predictor.predict(sequences)
```

### 3. VAE序列插值

```python
from src.advanced_vae_generator import VAEPeptideGenerator

vae = VAEPeptideGenerator()
vae.prepare_data()
vae.build_model()
vae.train_model()

# 在两个序列间插值
interpolated = vae.interpolate_sequences("KWKLF", "RRWWF", num_steps=5)
```

## 性能指标

### 模型性能 (5折交叉验证)
- **准确率**: 86.7% ± 2.1%
- **精确率**: 88.2% ± 1.8%
- **召回率**: 84.1% ± 2.4%
- **F1分数**: 86.1% ± 1.9%
- **AUC**: 0.921 ± 0.015

### 生成多样性
- **序列变异**: 高保守性，85%序列与种子序列相似度>0.7
- **理性设计**: 数据驱动的结构多样性，结合统计洞察与生物学约束
- **VAE生成**: 高创新性，60%序列为全新组合

## 常见问题与解决

### Q1: BioBERT模型下载失败
```bash
# 手动下载并放置到model/biobert/目录
```

### Q2: CUDA内存不足
```python
# 在代码开头添加
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用GPU 0
# 或者强制使用CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Q3: 特征维度不匹配
确保使用正确的预处理参数，与训练时保持一致：
- rawseq_maxlen: 20
- ngram_n: 3
- window_size: 5
- terminal_n: 3

## 引用与参考

如果您在研究中使用了本项目，请引用：

```bibtex
@software{amp_multimodal_discovery,
  title={AMP-MultiModal-Discovery: Intelligent Discovery System for Antimicrobial Peptides using Multi-modal Deep Learning},
  author={Your Name},
  year={2025},
  url={https://github.com/linkersea/AMP-MultiModal-Discovery}
}
```

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

**联系方式**: [dengs2021@163.com]  
**项目主页**: [https://github.com/linkersea/AMP-MultiModal-Discovery]
