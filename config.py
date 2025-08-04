"""
广告DSP多任务多场景模型配置文件
"""

# 特征定义
SPARSE_FEATURES = [
    'hour', 'weekday', 'adv_id', 'affiliate_id', 'campaign_id', 'ad_group_id', 'ad_id',
    'creative_id', 'feature_1', 'pos', 'instl', 'response_type', 'ad_format', 'os',
    'device_make', 'bundle_id', 'country', 'package', 'category', 'connection_type',
    'device_model', 'lang', 'publisher_id', 'first_ssp', 'last_ssp', 'video_placement',
    'is_rewarded', 'offer_id', 'supply_developer_id', 'supply_genreId', 'supply_version',
    'supply_minimum_os_version', 'supply_industry_id', 'is_oem', 'tag_id', 'osv', 'ua',
    'demand_developer_id', 'demand_genreId', 'demand_version', 'demand_minimum_os_version',
    'demand_industry_id', 'ip', 'device_id', 'ad_width', 'ad_height'
]

# 场景特征（supply开头特征和ip）
SCENARIO_FEATURES = [
    'supply_developer_id', 'supply_genreId', 'supply_version',
    'supply_minimum_os_version', 'supply_industry_id', 'ip'
]

# 任务定义
TASKS = ['ctr', 'cvr', 'ivr']

# 模型参数
MODEL_CONFIG = {
    'embedding_dim': 64,
    'hidden_dims': [512, 256, 128],
    'dropout_rate': 0.3,
    'batch_norm': True,
    'scenario_embedding_dim': 32,
    'tower_hidden_dims': [256, 128],
    'calibration_hidden_dim': 64,
}

# 训练参数
TRAINING_CONFIG = {
    'batch_size': 2048,
    'learning_rate': 0.001,
    'epochs': 100,
    'early_stopping_patience': 10,
    'weight_decay': 1e-5,
    'gradient_clip_norm': 1.0,
    'val_split': 0.2,
    'test_split': 0.1,
}

# 样本不均衡处理配置
IMBALANCE_CONFIG = {
    'use_focal_loss': True,
    'focal_alpha': [0.25, 0.5, 0.25],  # 对应ctr, cvr, ivr的权重
    'focal_gamma': 2.0,
    'use_class_weights': True,
    'oversample_minority': True,
    'undersample_majority': False,
}

# 多任务权重策略
MULTITASK_CONFIG = {
    'use_uncertainty_weighting': True,  # 使用不确定性权重
    'use_gradient_normalization': True,  # 梯度归一化
    'task_weights': [1.0, 1.0, 1.0],  # 初始任务权重
    'weight_update_frequency': 1000,  # 权重更新频率
}

# 校准配置
CALIBRATION_CONFIG = {
    'use_platt_scaling': True,
    'use_isotonic_regression': True,
    'calibration_bins': 10,
    'temperature_scaling': True,
}

# 正则化和防过拟合
REGULARIZATION_CONFIG = {
    'l1_lambda': 1e-6,
    'l2_lambda': 1e-5,
    'embedding_l2': 1e-6,
    'dropout_schedule': True,  # 训练过程中动态调整dropout
    'label_smoothing': 0.1,
    'mixup_alpha': 0.2,
}

# 优化器配置
OPTIMIZER_CONFIG = {
    'optimizer': 'AdamW',
    'scheduler': 'ReduceLROnPlateau',
    'scheduler_patience': 5,
    'scheduler_factor': 0.5,
    'min_lr': 1e-6,
}

# 数据处理配置
DATA_CONFIG = {
    'max_vocab_size': 10000,  # 每个特征的最大词汇表大小
    'min_frequency': 5,  # 特征值最小出现频率
    'use_hash_encoding': True,  # 对高基数特征使用hash编码
    'hash_bucket_size': 100000,
}

# 评估指标
METRICS = ['auc', 'logloss', 'accuracy', 'precision', 'recall', 'f1']

# 保存路径
PATHS = {
    'model_save_dir': './models',
    'log_dir': './logs',
    'tensorboard_dir': './runs',
    'feature_vocab_dir': './vocab',
    'checkpoint_dir': './checkpoints',
}