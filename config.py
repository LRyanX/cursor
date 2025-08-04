import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import yaml
import json

@dataclass
class ModelConfig:
    """模型配置"""
    feature_dim: int = 100
    num_scenes: int = 5
    num_experts: int = 6
    expert_hidden_dim: int = 256
    tower_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    tasks: List[str] = field(default_factory=lambda: ['ctr', 'cvr', 'ivr'])
    
    # 场景适配层配置
    scene_adaptation_hidden_dim: Optional[int] = None
    
    # 校准层配置
    calibration_bins: int = 10
    
    # Focal Loss配置
    focal_alpha: Dict[str, float] = field(default_factory=lambda: {
        'ctr': 1.0, 'cvr': 2.0, 'ivr': 3.0
    })
    focal_gamma: Dict[str, float] = field(default_factory=lambda: {
        'ctr': 2.0, 'cvr': 2.5, 'ivr': 3.0
    })

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    val_batch_size: int = 512
    
    # 学习率调度
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.5
    
    # 早停
    patience: int = 10
    
    # 梯度裁剪
    max_grad_norm: float = 1.0
    
    # 梯度平衡器参数
    gradient_balance_alpha: float = 0.5
    
    # 设备配置
    device: str = 'cuda'
    num_workers: int = 4
    pin_memory: bool = True
    
    # 日志和检查点
    log_dir: str = './logs'
    checkpoint_dir: str = './checkpoints'
    save_freq: int = 10
    
    # 混合精度训练
    use_amp: bool = True

@dataclass
class DataConfig:
    """数据配置"""
    # 数据文件路径
    data_path: Optional[str] = None
    
    # 特征和标签列配置
    feature_cols: List[str] = field(default_factory=list)
    target_cols: Dict[str, str] = field(default_factory=lambda: {
        'ctr': 'ctr', 'cvr': 'cvr', 'ivr': 'ivr'
    })
    scene_col: str = 'scene_id'
    
    # 数据分割
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    
    # 样本均衡策略
    balance_strategy: str = 'weighted'  # 'weighted', 'oversample', 'undersample', 'combine'
    
    # 数据预处理
    normalize_features: bool = True
    handle_missing: str = 'mean'  # 'mean', 'median', 'mode', 'drop'
    
    # 示例数据生成（用于测试）
    generate_sample_data: bool = False
    num_samples: int = 10000
    num_features: int = 100

@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment_name: str = 'multi_task_dsp'
    version: str = 'v1.0'
    description: str = 'Multi-task multi-scene DSP model'
    
    # 随机种子
    seed: int = 42
    
    # 实验跟踪
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    
    # 评估配置
    eval_metrics: List[str] = field(default_factory=lambda: [
        'auc', 'pr_auc', 'logloss', 'ece', 'mce'
    ])
    
    # 可视化
    plot_training_curves: bool = True
    plot_calibration: bool = True
    plot_scene_comparison: bool = True

@dataclass
class DSPConfig:
    """完整的DSP配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'DSPConfig':
        """从YAML文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {}))
        )
    
    @classmethod
    def from_json(cls, filepath: str) -> 'DSPConfig':
        """从JSON文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {}))
        )
    
    def to_yaml(self, filepath: str):
        """保存配置到YAML文件"""
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'experiment': self.experiment.__dict__
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def to_json(self, filepath: str):
        """保存配置到JSON文件"""
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'experiment': self.experiment.__dict__
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def validate(self) -> List[str]:
        """验证配置的有效性"""
        errors = []
        
        # 验证模型配置
        if self.model.feature_dim <= 0:
            errors.append("feature_dim must be positive")
        
        if self.model.num_scenes <= 0:
            errors.append("num_scenes must be positive")
        
        if self.model.num_experts <= 0:
            errors.append("num_experts must be positive")
        
        if not self.model.tasks:
            errors.append("tasks cannot be empty")
        
        # 验证训练配置
        if self.training.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        if self.training.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if self.training.num_epochs <= 0:
            errors.append("num_epochs must be positive")
        
        # 验证数据配置
        if self.data.test_size <= 0 or self.data.test_size >= 1:
            errors.append("test_size must be between 0 and 1")
        
        if self.data.val_size <= 0 or self.data.val_size >= 1:
            errors.append("val_size must be between 0 and 1")
        
        if self.data.balance_strategy not in ['weighted', 'oversample', 'undersample', 'combine']:
            errors.append("balance_strategy must be one of: weighted, oversample, undersample, combine")
        
        return errors

def create_default_config() -> DSPConfig:
    """创建默认配置"""
    return DSPConfig()

def create_sample_config() -> DSPConfig:
    """创建示例配置"""
    config = DSPConfig()
    
    # 模型配置
    config.model.feature_dim = 100
    config.model.num_scenes = 5
    config.model.tasks = ['ctr', 'cvr', 'ivr']
    
    # 训练配置
    config.training.num_epochs = 50
    config.training.learning_rate = 1e-3
    config.training.batch_size = 256
    
    # 数据配置
    config.data.generate_sample_data = True
    config.data.num_samples = 10000
    config.data.balance_strategy = 'weighted'
    
    # 实验配置
    config.experiment.experiment_name = 'dsp_sample_experiment'
    config.experiment.use_tensorboard = True
    
    return config

def load_config(filepath: str) -> DSPConfig:
    """自动检测文件格式并加载配置"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    _, ext = os.path.splitext(filepath)
    
    if ext.lower() in ['.yaml', '.yml']:
        return DSPConfig.from_yaml(filepath)
    elif ext.lower() == '.json':
        return DSPConfig.from_json(filepath)
    else:
        raise ValueError(f"Unsupported config file format: {ext}")

# 预定义的配置模板
CONFIG_TEMPLATES = {
    'development': {
        'model': {
            'feature_dim': 50,
            'num_scenes': 3,
            'num_experts': 4,
            'tasks': ['ctr', 'cvr']
        },
        'training': {
            'num_epochs': 10,
            'batch_size': 128,
            'learning_rate': 1e-3
        },
        'data': {
            'generate_sample_data': True,
            'num_samples': 1000,
            'balance_strategy': 'weighted'
        }
    },
    'production': {
        'model': {
            'feature_dim': 200,
            'num_scenes': 10,
            'num_experts': 8,
            'expert_hidden_dim': 512,
            'tower_hidden_dims': [512, 256, 128]
        },
        'training': {
            'num_epochs': 200,
            'batch_size': 512,
            'learning_rate': 5e-4,
            'use_amp': True
        },
        'data': {
            'balance_strategy': 'combine',
            'normalize_features': True
        }
    },
    'research': {
        'model': {
            'feature_dim': 100,
            'num_scenes': 5,
            'num_experts': 6,
            'tasks': ['ctr', 'cvr', 'ivr']
        },
        'training': {
            'num_epochs': 100,
            'batch_size': 256,
            'learning_rate': 1e-3,
            'patience': 15
        },
        'experiment': {
            'use_tensorboard': True,
            'plot_training_curves': True,
            'plot_calibration': True
        }
    }
}

def create_config_from_template(template_name: str) -> DSPConfig:
    """从模板创建配置"""
    if template_name not in CONFIG_TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(CONFIG_TEMPLATES.keys())}")
    
    template = CONFIG_TEMPLATES[template_name]
    config = DSPConfig()
    
    # 更新配置
    if 'model' in template:
        for key, value in template['model'].items():
            setattr(config.model, key, value)
    
    if 'training' in template:
        for key, value in template['training'].items():
            setattr(config.training, key, value)
    
    if 'data' in template:
        for key, value in template['data'].items():
            setattr(config.data, key, value)
    
    if 'experiment' in template:
        for key, value in template['experiment'].items():
            setattr(config.experiment, key, value)
    
    return config