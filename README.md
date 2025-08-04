# 广告DSP多任务多场景预估模型

一个高性能的广告DSP平台预估模型，支持多任务学习（CTR、CVR、IVR）和多场景适配，解决样本不均衡、跷跷板问题和指标校准等挑战。

## 🚀 核心特性

### 🎯 多任务学习
- **CTR（点击率）预估**：预测广告被点击的概率
- **CVR（转化率）预估**：预测点击后的转化概率  
- **IVR（曝光转化率）预估**：预测曝光后直接转化的概率
- **Cross-stitch机制**：任务间知识共享，避免跷跷板问题
- **不确定性权重学习**：自动平衡多任务损失

### 🌐 多场景适配
- **场景感知机制**：根据投放地理位置、平台等场景调整模型
- **场景门控网络**：为不同场景动态调整特征权重
- **场景特征嵌入**：专门处理场景相关特征（supply开头特征和ip）

### ⚖️ 样本不均衡处理
- **Focal Loss**：解决正负样本不均衡问题
- **类别权重**：自动计算并应用类别权重
- **多种采样策略**：SMOTE、ADASYN、组合采样等
- **加权随机采样**：训练时的动态采样策略

### 📏 指标校准
- **Platt Scaling**：逻辑回归校准
- **等渗回归**：非参数校准方法
- **温度缩放**：神经网络校准
- **分箱校准**：基于分箱的校准方法
- **校准指标**：ECE、MCE、Brier Score等

### 🛡️ 防过拟合机制
- **早停机制**：基于验证集性能自动停止训练
- **Dropout调度**：训练过程中动态调整Dropout率
- **L1/L2正则化**：权重正则化防止过拟合
- **标签平滑**：减少过度自信
- **梯度裁剪**：防止梯度爆炸

### ⚡ 性能优化
- **哈希编码**：处理高基数特征，减少内存占用
- **词汇表管理**：智能的特征编码和词汇表构建
- **批量处理**：高效的数据加载和批处理
- **混合精度训练**：支持FP16加速训练
- **多头注意力**：增强特征交互能力

## 🏗️ 模型架构

```
输入特征
    ↓
特征嵌入层 (Embedding)
    ↓
共享底层网络 (Shared Network)
    ↓
多头注意力 (Multi-Head Attention)
    ↓
场景门控机制 (Scenario Gates)
    ↓
Cross-stitch单元 (Cross-stitch)
    ↓
任务特定塔 (Task-specific Towers)
    ↓
校准层 (Calibration)
    ↓
最终预测 (CTR, CVR, IVR)
```

## 📦 项目结构

```
.
├── config.py                 # 配置文件
├── main.py                  # 主程序入口
├── trainer.py               # 训练器
├── requirements.txt         # 依赖包
├── README.md               # 项目文档
├── models/                 # 模型模块
│   ├── feature_processor.py  # 特征处理
│   └── multitask_model.py    # 多任务模型
└── utils/                  # 工具模块
    ├── data_utils.py        # 数据工具
    └── calibration.py       # 校准工具
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone <repository_url>
cd multitask-dsp-model

# 安装依赖
pip install -r requirements.txt
```

### 2. 快速训练（使用示例数据）

```bash
# 快速训练模式（适合测试）
python main.py --train_mode fast --use_sample_data

# 完整训练模式
python main.py --train_mode full --use_sample_data
```

### 3. 使用自己的数据

```bash
# 使用CSV文件训练
python main.py --data_path your_data.csv --output_dir ./your_output
```

### 4. 数据格式要求

CSV文件需要包含以下列：
- **特征列**：所有在`config.py`中定义的稀疏特征
- **目标列**：`ctr`, `cvr`, `ivr` (0或1的二分类标签)

示例数据格式：
```csv
hour,weekday,adv_id,country,ctr,cvr,ivr
10,1,123,US,1,0,0
14,2,456,CN,0,0,0
...
```

## ⚙️ 配置说明

主要配置项在`config.py`中：

### 特征配置
```python
SPARSE_FEATURES = [
    'hour', 'weekday', 'adv_id', 'country', ...
]

SCENARIO_FEATURES = [
    'supply_developer_id', 'supply_genreId', 'ip', ...
]
```

### 模型配置
```python
MODEL_CONFIG = {
    'embedding_dim': 64,          # 嵌入维度
    'hidden_dims': [512, 256, 128],  # 隐藏层维度
    'dropout_rate': 0.3,          # Dropout率
    'scenario_embedding_dim': 32,  # 场景嵌入维度
}
```

### 训练配置
```python
TRAINING_CONFIG = {
    'batch_size': 2048,
    'learning_rate': 0.001,
    'epochs': 100,
    'early_stopping_patience': 10,
}
```

## 📊 模型特色功能

### 1. 样本不均衡处理

模型自动检测样本不均衡情况并应用相应策略：

```python
# Focal Loss参数
IMBALANCE_CONFIG = {
    'use_focal_loss': True,
    'focal_alpha': [0.25, 0.5, 0.25],  # CTR, CVR, IVR权重
    'focal_gamma': 2.0,
}
```

### 2. 多任务权重自适应

使用不确定性权重自动平衡多任务学习：

```python
# 启用不确定性权重
MULTITASK_CONFIG = {
    'use_uncertainty_weighting': True,
    'use_gradient_normalization': True,
}
```

### 3. 指标校准

支持多种校准方法确保预测值接近真实概率：

```python
# 校准配置
CALIBRATION_CONFIG = {
    'use_platt_scaling': True,
    'use_isotonic_regression': True,
    'temperature_scaling': True,
}
```

## 📈 性能监控

### TensorBoard可视化

训练过程自动记录到TensorBoard：

```bash
# 启动TensorBoard
tensorboard --logdir ./outputs/logs
```

可以监控：
- 各任务的训练/验证损失
- 模型性能指标（AUC、准确率等）
- 学习率变化
- 梯度信息

### 校准效果可视化

模型自动生成校准曲线图，评估预测概率的可靠性。

## 🔧 高级功能

### 1. 数据增强

支持多种数据增强技术：
- **Mixup**：特征和标签的线性插值
- **特征Dropout**：随机丢弃部分特征
- **噪声注入**：添加高斯噪声

### 2. 梯度归一化

解决多任务训练中的跷跷板问题：
- 计算各任务的梯度范数
- 动态调整任务权重
- 防止某个任务主导训练

### 3. 动态学习率调度

支持多种学习率调度策略：
- **ReduceLROnPlateau**：基于验证性能
- **CosineAnnealing**：余弦退火
- **StepLR**：阶梯式衰减

## 🎯 模型评估

### 评估指标

自动计算多种评估指标：
- **AUC**：ROC曲线下面积
- **准确率**：分类准确性
- **精确率/召回率/F1**：综合性能评估
- **Log Loss**：概率预测质量

### 校准指标

评估预测概率的可靠性：
- **ECE (Expected Calibration Error)**：期望校准误差
- **MCE (Maximum Calibration Error)**：最大校准误差
- **Brier Score**：概率预测得分

## 🚨 注意事项

### 1. 内存优化
- 使用哈希编码处理高基数特征
- 合理设置batch_size避免OOM
- 启用梯度检查点节省内存

### 2. 训练稳定性
- 使用梯度裁剪防止梯度爆炸
- 启用早停避免过拟合
- 定期保存检查点

### 3. 数据质量
- 确保特征值编码正确
- 处理缺失值和异常值
- 验证目标变量分布

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢以下开源项目的启发：
- PyTorch深度学习框架
- scikit-learn机器学习库
- imbalanced-learn样本不均衡处理
- TensorBoard可视化工具

---

**如有问题，请提交Issue或联系项目维护者。**
