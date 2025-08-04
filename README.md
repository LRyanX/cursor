# 多任务多场景DSP广告预估模型

这是一个专为DSP（Demand Side Platform）广告平台设计的多任务多场景预估模型，能够同时预测点击率（CTR）、转化率（CVR）和曝光转化率（IVR），并适应不同的投放场景。

## 🚀 核心特性

### 1. 多任务学习架构
- **MMoE (Multi-gate Mixture-of-Experts)**: 使用专家混合网络处理任务间的关系
- **任务特定塔网络**: 为每个任务学习专门的表示
- **动态权重平衡**: 自动调整任务权重，解决跷跷板问题

### 2. 多场景适配
- **场景感知注意力机制**: 自动学习不同场景的特征重要性
- **场景特定变换**: 为每个场景学习独立的特征变换
- **跨场景知识共享**: 在保持场景特异性的同时共享通用知识

### 3. 样本不均衡处理
- **多种重采样策略**: SMOTE、欠采样、组合采样
- **Focal Loss**: 专门处理类别不均衡的损失函数
- **样本权重**: 基于类别分布的动态权重调整

### 4. 指标校准
- **Platt Scaling**: 可学习的校准参数
- **校准评估**: ECE (Expected Calibration Error) 和 MCE (Maximum Calibration Error)
- **预测置信度对齐**: 确保模型预测值与真实概率一致

## 📊 模型架构

```
输入特征 → 特征嵌入 → 场景适配层 → MMoE专家网络 → 任务塔网络 → 校准层 → 输出
```

### 核心组件说明

1. **SceneAdaptationLayer**: 为不同场景学习特定的特征变换
2. **MMOEExpertNet**: 多个专家网络学习不同的特征表示
3. **GatingNetwork**: 为每个任务分配专家权重
4. **MultiTaskTower**: 任务特定的深度网络
5. **CalibrationLayer**: 使用Platt Scaling进行预测校准
6. **FocalLoss**: 处理样本不均衡的损失函数
7. **GradientBalancer**: 动态调整任务权重避免跷跷板问题

## 🛠️ 安装和使用

### 环境要求

- Python 3.8+
- PyTorch 1.12+
- CUDA (可选，用于GPU训练)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 快速开始

#### 1. 使用示例数据训练

```bash
# 使用默认配置训练
python main.py --template development

# 使用研究配置训练
python main.py --template research

# 使用生产配置训练
python main.py --template production
```

#### 2. 使用自定义数据

```bash
# 准备配置文件 config.yaml
python main.py --config config.yaml --mode train
```

#### 3. 仅评估模型

```bash
python main.py --mode eval --checkpoint ./outputs/checkpoints/best_model.pth
```

### 配置文件示例

```yaml
model:
  feature_dim: 100
  num_scenes: 5
  num_experts: 6
  expert_hidden_dim: 256
  tower_hidden_dims: [256, 128]
  tasks: ["ctr", "cvr", "ivr"]

training:
  num_epochs: 100
  learning_rate: 0.001
  batch_size: 256
  patience: 10
  device: "cuda"

data:
  data_path: "./data/train.csv"
  balance_strategy: "weighted"
  test_size: 0.2
  val_size: 0.1

experiment:
  experiment_name: "dsp_experiment"
  use_tensorboard: true
  plot_training_curves: true
```

## 📁 项目结构

```
├── multi_task_dsp_model.py    # 主模型定义
├── data_utils.py              # 数据处理工具
├── trainer.py                 # 训练和评估逻辑
├── config.py                  # 配置管理
├── main.py                    # 主程序入口
├── requirements.txt           # 依赖包列表
└── README.md                  # 项目说明
```

## 🔧 核心功能详解

### 多任务学习

模型同时训练三个相关任务：

- **CTR (Click-Through Rate)**: 点击率预测
- **CVR (Conversion Rate)**: 转化率预测（基于点击）
- **IVR (Impression-to-Conversion Rate)**: 曝光转化率预测

通过共享底层表示和任务特定的塔网络，模型能够：
- 利用任务间的相关性提升整体性能
- 避免负迁移和跷跷板效应
- 处理不同任务的数据稀疏性问题

### 场景适配

支持多种投放场景（如地理位置、平台类型等）：

- **注意力机制**: 自动学习场景特征重要性
- **场景特定变换**: 为每个场景学习独立参数
- **知识共享**: 在场景间共享通用特征表示

### 样本不均衡处理

提供多种策略处理广告数据中的样本不均衡：

1. **加权策略** (`weighted`): 基于类别分布计算样本权重
2. **过采样** (`oversample`): 使用SMOTE生成合成样本
3. **欠采样** (`undersample`): 随机减少多数类样本
4. **组合策略** (`combine`): 结合过采样和欠采样

### 梯度平衡

动态调整任务权重解决多任务学习中的跷跷板问题：

- 监控各任务的损失下降速度
- 自动增加学习困难任务的权重
- 防止某个任务主导整个训练过程

### 指标校准

确保模型预测概率的可靠性：

- **Platt Scaling**: 学习校准参数 α 和 β
- **校准评估**: 计算 ECE 和 MCE 指标
- **可视化**: 绘制校准曲线分析预测质量

## 📈 评估指标

模型使用以下指标评估性能：

- **AUC**: ROC曲线下面积
- **PR-AUC**: Precision-Recall曲线下面积
- **Log Loss**: 交叉熵损失
- **ECE**: 期望校准误差
- **MCE**: 最大校准误差

支持整体评估和分场景评估，确保模型在各种条件下都能保持良好性能。

## 🎯 最佳实践

### 数据准备

1. **特征工程**: 确保特征质量和相关性
2. **数据清洗**: 处理缺失值和异常值
3. **场景标识**: 明确定义场景分类标准

### 模型调优

1. **专家数量**: 根据数据复杂度调整专家网络数量
2. **网络深度**: 平衡模型容量和过拟合风险
3. **学习率**: 使用学习率调度获得更好收敛

### 训练策略

1. **早停**: 防止过拟合，提升泛化能力
2. **梯度裁剪**: 稳定训练过程
3. **权重平衡**: 根据业务需求调整任务重要性

## 🔬 实验和监控

### TensorBoard可视化

```bash
tensorboard --logdir ./outputs/logs
```

可以查看：
- 训练和验证损失曲线
- 各任务的AUC变化
- 校准误差趋势
- 学习率调度

### 输出文件

训练完成后，在输出目录中可以找到：

- `config.yaml/json`: 完整配置文件
- `training_history.json`: 训练历史
- `training_curves.png`: 训练曲线图
- `eval_results.json`: 评估结果
- `checkpoints/`: 模型检查点
- `logs/`: 训练日志

## 🤝 贡献指南

欢迎提交问题和功能请求！如果您想为项目做出贡献：

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 创建 GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 这是一个研究项目，在生产环境使用前请进行充分测试和验证。
