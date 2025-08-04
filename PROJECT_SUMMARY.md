# 广告DSP多任务学习模型 - 项目总结

## 🎯 项目概述

本项目为广告DSP平台设计了一个完整的多任务学习模型，能够同时预测CTR（点击率）、CVR（转化率）和IVR（曝光转化率），并解决了样本不均衡、跷跷板问题、指标校准等关键挑战。

## ✅ 已完成功能

### 1. 核心模型架构
- **多任务学习模型** (`model.py`)
  - 共享特征提取器
  - 场景适配器
  - 任务特定头部
  - 注意力机制
  - 任务权重自适应学习

### 2. 数据处理模块 (`data_processor.py`)
- **特征处理器**：处理稀疏特征、数值特征、类别特征
- **场景嵌入**：为不同场景创建嵌入表示
- **样本权重平衡**：解决样本不均衡问题
- **数据加载器**：支持训练/验证/测试集分割

### 3. 损失函数和优化 (`utils.py`)
- **Focal Loss**：处理正样本稀疏问题
- **Pareto Loss**：避免跷跷板问题
- **样本权重平衡**：根据正负样本比例调整权重
- **评估指标**：AUC、Log Loss、Brier Score、校准误差

### 4. 模型校准 (`utils.py`)
- **多种校准方法**：Isotonic、Platt、Temperature Scaling
- **校准曲线可视化**：直观展示校准效果
- **真实值接近性**：确保预估值与真实值基本接近

### 5. 训练和评估 (`main.py`)
- **完整训练流程**：支持单模型和集成模型训练
- **超参数优化**：使用Optuna进行自动调参
- **早停机制**：防止过拟合
- **混合精度训练**：加速训练过程
- **特征重要性分析**：识别重要特征

### 6. 配置管理 (`config.py`)
- **模型配置**：网络结构、训练参数、损失函数配置
- **特征配置**：特征类型、场景特征、特征工程设置
- **训练配置**：优化器、调度器、正则化设置

## 📊 特征支持

### 稀疏特征 (45个)
```
hour, weekday, adv_id, affiliate_id, campaign_id, ad_group_id, ad_id, 
creative_id, feature_1, pos, instl, response_type, ad_format, os, 
device_make, bundle_id, country, package, category, connection_type,
device_model, lang, publisher_id, first_ssp, last_ssp, video_placement,
is_rewarded, offer_id, supply_developer_id, supply_genreId, supply_version,
supply_minimum_os_version, supply_industry_id, is_oem, tag_id, osv, ua,
demand_developer_id, demand_genreId, demand_version, demand_minimum_os_version,
demand_industry_id, ip, device_id, ad_width, ad_height
```

### 场景特征 (6个)
```
supply_developer_id, supply_genreId, supply_version,
supply_minimum_os_version, supply_industry_id, ip
```

## 🚀 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 快速测试
```bash
python3 test_basic.py
```

### 3. 运行示例
```bash
python3 example.py
```

### 4. 完整训练
```bash
# 使用示例数据
python3 main.py --use_sample_data --calibrate --feature_importance

# 使用真实数据
python3 main.py --data_path your_data.csv --calibrate --feature_importance

# 集成模型训练
python3 main.py --use_sample_data --ensemble --num_models 3 --calibrate

# 超参数优化
python3 main.py --use_sample_data --optimize_hyperparams --n_trials 50
```

## 🎯 核心算法

### 1. Pareto多任务学习
```python
# 避免跷跷板问题的损失函数
precision = torch.exp(-log_vars[i])
precision_loss = precision * focal_loss + log_vars[i]
total_loss = sum(precision_losses.values())
```

### 2. 场景适配
```python
# 场景分类和嵌入
scenario_probs = F.softmax(scenario_logits, dim=1)
weighted_scenario_emb = torch.matmul(scenario_probs, scenario_emb)
adapted_features = scenario_projector(weighted_scenario_emb)
```

### 3. 样本权重平衡
```python
# 根据正负样本比例调整权重
pos_ratio = y.mean()
if pos_ratio > 0:
    pos_weight_adjusted = pos_weight / pos_ratio
    weights[y == 1] = pos_weight_adjusted
```

## 📈 性能优化

### 1. 计算优化
- 混合精度训练加速
- 梯度累积减少内存使用
- 多进程数据加载

### 2. 内存优化
- 动态批次大小
- 梯度检查点
- 模型量化

### 3. 训练优化
- 早停机制防止过拟合
- 学习率调度
- 梯度裁剪稳定训练

## 🔧 配置说明

### 模型配置
- `embedding_dim`: 嵌入维度 (16)
- `hidden_dims`: 隐藏层维度 ([256, 128, 64])
- `dropout_rate`: Dropout比率 (0.3)
- `learning_rate`: 学习率 (1e-3)
- `batch_size`: 批次大小 (1024)
- `num_epochs`: 训练轮数 (100)

### 特征配置
- `sparse_features`: 稀疏特征列表 (45个特征)
- `scenario_features`: 场景特征列表 (6个特征)
- `feature_engineering`: 是否进行特征工程 (True)
- `feature_selection`: 是否进行特征选择 (True)

### 训练配置
- `optimizer`: 优化器类型 ("adam")
- `scheduler`: 学习率调度器 ("cosine")
- `use_amp`: 是否使用混合精度训练 (True)
- `gradient_clip`: 梯度裁剪阈值 (1.0)

## 📊 输出结果

训练完成后，会在输出目录生成以下文件：

### 模型文件
- `models/best_model.pth` - 最佳模型权重
- `models/training_history.pkl` - 训练历史
- `models/ensemble_model_*.pth` - 集成模型权重

### 结果文件
- `results/test_metrics.csv` - 测试集评估指标
- `results/calibration_curves.png` - 校准曲线图
- `results/training_curves.png` - 训练曲线图
- `results/feature_importance.csv` - 特征重要性
- `results/feature_importance.png` - 特征重要性图

### 日志文件
- `training.log` - 训练日志

## 🎯 解决的问题

### 1. 样本不均衡
- ✅ 使用Focal Loss处理正样本稀疏问题
- ✅ 样本权重平衡机制
- ✅ 过采样策略

### 2. 跷跷板问题
- ✅ Pareto最优损失函数
- ✅ 任务权重自适应学习
- ✅ 注意力机制处理任务间交互

### 3. 指标校准
- ✅ 多种校准方法支持
- ✅ 校准曲线可视化
- ✅ 真实值接近性保证

### 4. 场景适配
- ✅ 场景嵌入学习
- ✅ 场景分类器
- ✅ 场景特定特征投影

### 5. 性能优化
- ✅ 混合精度训练
- ✅ 早停机制
- ✅ 梯度裁剪
- ✅ 特征选择

## 🔍 故障排除

### 常见问题

1. **内存不足**
   - 减少batch_size
   - 使用梯度累积
   - 启用混合精度训练

2. **训练不收敛**
   - 调整学习率
   - 检查数据预处理
   - 增加正则化

3. **过拟合**
   - 增加dropout_rate
   - 减少模型复杂度
   - 增加训练数据

4. **跷跷板问题**
   - 调整任务权重
   - 使用Pareto损失函数
   - 平衡数据分布

## 📝 项目文件说明

| 文件 | 功能 | 大小 |
|------|------|------|
| `config.py` | 配置文件 | 5.0KB |
| `utils.py` | 工具函数 | 9.4KB |
| `data_processor.py` | 数据处理 | 15KB |
| `model.py` | 模型定义 | 17KB |
| `main.py` | 主程序 | 13KB |
| `example.py` | 示例脚本 | 12KB |
| `requirements.txt` | 依赖包 | 263B |
| `README.md` | 项目文档 | 6.7KB |
| `test_basic.py` | 基础测试 | 6.8KB |
| `test_quick.py` | 快速测试 | 7.0KB |

## 🎉 总结

本项目成功实现了一个完整的广告DSP多任务学习模型，具备以下特点：

1. **功能完整**：支持CTR、CVR、IVR三个任务的联合预测
2. **技术先进**：使用Pareto多任务学习、场景适配、Focal Loss等先进技术
3. **性能优化**：混合精度训练、早停机制、梯度裁剪等优化措施
4. **易于使用**：提供完整的训练、评估、可视化流程
5. **文档完善**：详细的README和示例代码

该模型能够有效解决广告DSP平台中的关键挑战，为实际业务应用提供了可靠的技术基础。