"""
广告DSP多任务学习模型配置文件
"""
import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ModelConfig:
    """模型配置"""
    # 基础配置
    seed: int = 42
    device: str = "cuda" if os.path.exists("/dev/cuda") else "cpu"
    
    # 模型架构
    embedding_dim: int = 16
    hidden_dims: List[int] = None
    dropout_rate: float = 0.3
    batch_norm: bool = True
    
    # 多任务配置
    tasks: List[str] = None
    task_weights: Dict[str, float] = None
    
    # 场景配置
    scenario_features: List[str] = None
    scenario_embedding_dim: int = 8
    
    # 训练配置
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 1024
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    # 损失函数配置
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # 采样配置
    sample_weights: Dict[str, float] = None
    oversample_ratio: float = 1.0
    
    # 校准配置
    calibration_method: str = "isotonic"  # isotonic, platt, temperature
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]
        
        if self.tasks is None:
            self.tasks = ["ctr", "cvr", "ivr"]
        
        if self.task_weights is None:
            self.task_weights = {"ctr": 1.0, "cvr": 1.0, "ivr": 1.0}
        
        if self.scenario_features is None:
            self.scenario_features = [
                'supply_developer_id', 'supply_genreId', 'supply_version',
                'supply_minimum_os_version', 'supply_industry_id', 'ip'
            ]
        
        if self.sample_weights is None:
            self.sample_weights = {"ctr": 1.0, "cvr": 1.0, "ivr": 1.0}

@dataclass
class FeatureConfig:
    """特征配置"""
    # 稀疏特征
    sparse_features: List[str] = None
    
    # 数值特征
    numeric_features: List[str] = None
    
    # 类别特征
    categorical_features: List[str] = None
    
    # 场景特征
    scenario_features: List[str] = None
    
    # 特征工程
    feature_engineering: bool = True
    feature_selection: bool = True
    
    def __post_init__(self):
        if self.sparse_features is None:
            self.sparse_features = [
                'hour', 'weekday', 'adv_id', 'affiliate_id', 'campaign_id', 
                'ad_group_id', 'ad_id', 'creative_id', 'feature_1', 'pos', 
                'instl', 'response_type', 'ad_format', 'os', 'device_make', 
                'bundle_id', 'country', 'package', 'category', 'connection_type',
                'device_model', 'lang', 'publisher_id', 'first_ssp', 'last_ssp', 
                'video_placement', 'is_rewarded', 'offer_id', 'supply_developer_id',
                'supply_genreId', 'supply_version', 'supply_minimum_os_version',
                'supply_industry_id', 'is_oem', 'tag_id', 'osv', 'ua',
                'demand_developer_id', 'demand_genreId', 'demand_version',
                'demand_minimum_os_version', 'demand_industry_id', 'ip', 
                'device_id', 'ad_width', 'ad_height'
            ]
        
        if self.numeric_features is None:
            self.numeric_features = ['ad_width', 'ad_height', 'hour', 'weekday']
        
        if self.categorical_features is None:
            self.categorical_features = [
                'adv_id', 'affiliate_id', 'campaign_id', 'ad_group_id', 'ad_id',
                'creative_id', 'pos', 'instl', 'response_type', 'ad_format', 'os',
                'device_make', 'bundle_id', 'country', 'package', 'category',
                'connection_type', 'device_model', 'lang', 'publisher_id',
                'first_ssp', 'last_ssp', 'video_placement', 'is_rewarded',
                'offer_id', 'supply_developer_id', 'supply_genreId', 'supply_version',
                'supply_minimum_os_version', 'supply_industry_id', 'is_oem',
                'tag_id', 'osv', 'ua', 'demand_developer_id', 'demand_genreId',
                'demand_version', 'demand_minimum_os_version', 'demand_industry_id',
                'ip', 'device_id'
            ]
        
        if self.scenario_features is None:
            self.scenario_features = [
                'supply_developer_id', 'supply_genreId', 'supply_version',
                'supply_minimum_os_version', 'supply_industry_id', 'ip'
            ]

@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据配置
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # 优化器配置
    optimizer: str = "adam"  # adam, sgd, adagrad
    scheduler: str = "cosine"  # cosine, step, plateau
    
    # 正则化
    l1_reg: float = 0.0
    l2_reg: float = 1e-5
    
    # 梯度裁剪
    gradient_clip: float = 1.0
    
    # 混合精度训练
    use_amp: bool = True
    
    # 多GPU训练
    use_ddp: bool = False
    
    # 日志配置
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000

# 创建配置实例
model_config = ModelConfig()
feature_config = FeatureConfig()
training_config = TrainingConfig()