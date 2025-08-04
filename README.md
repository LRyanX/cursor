# 广告DSP多任务学习模型

这是一个专为广告DSP平台设计的多任务学习模型，能够同时预测CTR（点击率）、CVR（转化率）和IVR（曝光转化率），并解决样本不均衡、跷跷板问题、指标校准等关键挑战。

## 🚀 核心特性

### 1. 多任务学习架构
- **同时预测三个指标**：CTR、CVR、IVR
- **Pareto最优损失函数**：避免跷跷板问题
- **任务权重自适应学习**：动态平衡各任务性能

### 2. 场景适配能力
- **多场景支持**：不同地理位置、平台、设备等场景
- **场景嵌入学习**：自动学习场景特定特征
- **场景分类器**：智能识别和适配不同场景

### 3. 样本不均衡处理
- **Focal Loss**：处理正样本稀疏问题
- **样本权重平衡**：根据正负样本比例调整权重
- **过采样策略**：针对不同任务的数据规模差异

### 4. 指标校准
- **多种校准方法**：Isotonic、Platt、Temperature Scaling
- **校准曲线可视化**：直观展示校准效果
- **真实值接近性**：确保预估值与真实值基本接近

### 5. 性能优化
- **混合精度训练**：加速训练过程
- **早停机制**：防止过拟合
- **梯度裁剪**：稳定训练过程
- **特征选择**：减少计算复杂度

## 📁 项目结构

```
├── config.py              # 配置文件
├── utils.py               # 工具函数
├── data_processor.py      # 数据处理模块
├── model.py              # 模型定义
├── main.py               # 主程序
├── requirements.txt       # 依赖包
└── README.md             # 项目文档
```

## 🛠️ 安装依赖

```bash
pip install -r requirements.txt
```

## 📊 特征说明

### 稀疏特征 (Sparse Features)
包含以下45个特征：
- `hour`, `weekday` - 时间特征
- `adv_id`, `affiliate_id`, `campaign_id`, `ad_group_id`, `ad_id`, `creative_id` - 广告相关ID
- `feature_1`, `pos`, `instl`, `response_type`, `ad_format` - 广告展示特征
- `os`, `device_make`, `bundle_id`, `country`, `package`, `category` - 设备和应用特征
- `connection_type`, `device_model`, `lang`, `publisher_id` - 网络和设备特征
- `first_ssp`, `last_ssp`, `video_placement`, `is_rewarded`, `offer_id` - SSP和视频特征
- `supply_*` 系列特征 - 供应方特征（场景特征）
- `demand_*` 系列特征 - 需求方特征
- `ip`, `device_id`, `ad_width`, `ad_height` - 其他特征

### 场景特征 (Scenario Features)
- `supply_developer_id` - 供应方开发者ID
- `supply_genreId` - 供应方游戏类型ID
- `supply_version` - 供应方版本
- `supply_minimum_os_version` - 供应方最低OS版本
- `supply_industry_id` - 供应方行业ID
- `ip` - IP地址

## 🚀 使用方法

### 1. 使用示例数据训练

```bash
python main.py --use_sample_data --sample_size 10000 --calibrate --feature_importance
```

### 2. 使用真实数据训练

```bash
python main.py --data_path your_data.csv --calibrate --feature_importance
```

### 3. 集成模型训练

```bash
python main.py --use_sample_data --ensemble --num_models 3 --calibrate
```

### 4. 超参数优化

```bash
python main.py --use_sample_data --optimize_hyperparams --n_trials 50
```

### 5. 完整实验

```bash
python main.py --use_sample_data --ensemble --calibrate --feature_importance --optimize_hyperparams
```

## 📈 模型架构

### 1. 共享特征提取器
- 多层感知机处理稀疏特征
- Batch Normalization和Dropout防止过拟合
- ReLU激活函数

### 2. 场景适配器
- 场景嵌入学习
- 场景分类器
- 场景特定特征投影

### 3. 任务特定头部
- 每个任务独立的预测网络
- 注意力机制处理任务间交互
- 任务权重自适应学习

### 4. 损失函数
- **Focal Loss**：处理样本不均衡
- **Pareto Loss**：避免跷跷板问题
- **L2正则化**：防止过拟合

## 🎯 核心算法

### 1. Pareto多任务学习
```python
# Pareto最优损失函数
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

## 📊 评估指标

- **AUC**：ROC曲线下面积
- **Log Loss**：对数损失
- **Brier Score**：概率预测准确性
- **Calibration Error**：校准误差

## 🔧 配置说明

### 模型配置 (config.py)
- `embedding_dim`: 嵌入维度
- `hidden_dims`: 隐藏层维度
- `dropout_rate`: Dropout比率
- `learning_rate`: 学习率
- `batch_size`: 批次大小
- `num_epochs`: 训练轮数

### 特征配置
- `sparse_features`: 稀疏特征列表
- `scenario_features`: 场景特征列表
- `feature_engineering`: 是否进行特征工程
- `feature_selection`: 是否进行特征选择

### 训练配置
- `optimizer`: 优化器类型
- `scheduler`: 学习率调度器
- `use_amp`: 是否使用混合精度训练
- `gradient_clip`: 梯度裁剪阈值

## 📈 输出结果

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

## 🎯 性能优化

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

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

MIT License

## 📞 联系方式

如有问题，请提交Issue或联系项目维护者。
