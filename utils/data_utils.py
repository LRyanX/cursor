"""
数据处理工具：样本不均衡处理、数据增强、批次处理
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import random


class ImbalanceHandler:
    """样本不均衡处理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.use_focal_loss = config.get('use_focal_loss', True)
        self.focal_alpha = config.get('focal_alpha', [0.25, 0.5, 0.25])
        self.focal_gamma = config.get('focal_gamma', 2.0)
        self.use_class_weights = config.get('use_class_weights', True)
        self.oversample_minority = config.get('oversample_minority', True)
        self.undersample_majority = config.get('undersample_majority', False)
        
        self.class_weights = {}
        self.samplers = {}
        
    def compute_class_weights(self, targets: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """计算各任务的类别权重"""
        
        for task_name, target in targets.items():
            unique_classes = np.unique(target)
            if len(unique_classes) == 2:  # 二分类
                class_weight = compute_class_weight(
                    'balanced', 
                    classes=unique_classes, 
                    y=target
                )
                self.class_weights[task_name] = {
                    'negative': class_weight[0],
                    'positive': class_weight[1]
                }
            else:
                print(f"警告: 任务 {task_name} 不是二分类任务")
        
        return self.class_weights
    
    def get_sampling_strategy(self, target: np.ndarray, task_name: str) -> Dict:
        """获取采样策略"""
        
        unique, counts = np.unique(target, return_counts=True)
        total_samples = len(target)
        
        positive_ratio = counts[1] / total_samples if len(unique) == 2 else 0.5
        
        # 根据正样本比例调整采样策略
        if positive_ratio < 0.1:  # 极度不均衡
            sampling_strategy = {0: int(counts[1] * 3), 1: counts[1]}  # 3:1
        elif positive_ratio < 0.3:  # 中度不均衡
            sampling_strategy = {0: int(counts[1] * 2), 1: counts[1]}  # 2:1
        else:  # 轻度不均衡
            sampling_strategy = 'auto'
        
        return sampling_strategy
    
    def apply_oversampling(self, features: np.ndarray, targets: Dict[str, np.ndarray],
                          method: str = 'smote') -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """应用过采样技术"""
        
        resampled_features = features.copy()
        resampled_targets = {}
        
        for task_name, target in targets.items():
            sampling_strategy = self.get_sampling_strategy(target, task_name)
            
            if method == 'smote':
                sampler = SMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=42,
                    k_neighbors=min(5, np.sum(target == 1) - 1)  # 确保邻居数不超过少数类样本数
                )
            elif method == 'borderline_smote':
                sampler = BorderlineSMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=42
                )
            elif method == 'adasyn':
                sampler = ADASYN(
                    sampling_strategy=sampling_strategy,
                    random_state=42
                )
            else:
                raise ValueError(f"Unknown oversampling method: {method}")
            
            try:
                resampled_features, resampled_target = sampler.fit_resample(resampled_features, target)
                resampled_targets[task_name] = resampled_target
                print(f"任务 {task_name} 过采样完成: {len(resampled_target)} 样本")
                
            except ValueError as e:
                print(f"任务 {task_name} 过采样失败: {e}")
                resampled_targets[task_name] = target
        
        return resampled_features, resampled_targets
    
    def apply_undersampling(self, features: np.ndarray, targets: Dict[str, np.ndarray],
                           method: str = 'tomek') -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """应用欠采样技术"""
        
        resampled_features = features.copy()
        resampled_targets = {}
        
        for task_name, target in targets.items():
            if method == 'tomek':
                sampler = TomekLinks()
            elif method == 'enn':
                sampler = EditedNearestNeighbours()
            else:
                raise ValueError(f"Unknown undersampling method: {method}")
            
            try:
                resampled_features, resampled_target = sampler.fit_resample(resampled_features, target)
                resampled_targets[task_name] = resampled_target
                print(f"任务 {task_name} 欠采样完成: {len(resampled_target)} 样本")
                
            except ValueError as e:
                print(f"任务 {task_name} 欠采样失败: {e}")
                resampled_targets[task_name] = target
        
        return resampled_features, resampled_targets
    
    def apply_combined_sampling(self, features: np.ndarray, targets: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """应用组合采样技术"""
        
        resampled_features = features.copy()
        resampled_targets = {}
        
        for task_name, target in targets.items():
            sampling_strategy = self.get_sampling_strategy(target, task_name)
            
            # SMOTE + Tomek Links
            sampler = SMOTETomek(
                sampling_strategy=sampling_strategy,
                random_state=42
            )
            
            try:
                resampled_features, resampled_target = sampler.fit_resample(resampled_features, target)
                resampled_targets[task_name] = resampled_target
                print(f"任务 {task_name} 组合采样完成: {len(resampled_target)} 样本")
                
            except ValueError as e:
                print(f"任务 {task_name} 组合采样失败: {e}")
                resampled_targets[task_name] = target
        
        return resampled_features, resampled_targets
    
    def create_weighted_sampler(self, targets: Dict[str, np.ndarray], task_name: str) -> WeightedRandomSampler:
        """创建加权随机采样器"""
        
        target = targets[task_name]
        class_counts = np.bincount(target.astype(int))
        class_weights = 1.0 / class_counts
        
        # 为每个样本分配权重
        sample_weights = [class_weights[int(label)] for label in target]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )


class DataAugmentation:
    """数据增强工具"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.mixup_alpha = config.get('mixup_alpha', 0.2)
        self.cutmix_alpha = config.get('cutmix_alpha', 1.0)
        self.noise_level = config.get('noise_level', 0.01)
        
    def mixup(self, features: torch.Tensor, targets: Dict[str, torch.Tensor], 
              alpha: float = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Mixup数据增强"""
        
        if alpha is None:
            alpha = self.mixup_alpha
            
        batch_size = features.size(0)
        
        # 生成混合权重
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        # 随机置换索引
        index = torch.randperm(batch_size)
        
        # 混合特征
        mixed_features = lam * features + (1 - lam) * features[index]
        
        # 混合标签
        mixed_targets = {}
        for task_name, target in targets.items():
            mixed_targets[task_name] = lam * target + (1 - lam) * target[index]
        
        return mixed_features, mixed_targets
    
    def add_noise(self, features: torch.Tensor, noise_level: float = None) -> torch.Tensor:
        """添加高斯噪声"""
        
        if noise_level is None:
            noise_level = self.noise_level
            
        noise = torch.randn_like(features) * noise_level
        return features + noise
    
    def feature_dropout(self, features: Dict[str, torch.Tensor], 
                       drop_prob: float = 0.1) -> Dict[str, torch.Tensor]:
        """特征dropout增强"""
        
        augmented_features = {}
        
        for feature_name, feature_tensor in features.items():
            if random.random() < drop_prob:
                # 随机将某些特征设为<UNK>
                mask = torch.rand_like(feature_tensor.float()) < 0.1
                augmented_features[feature_name] = torch.where(mask, 0, feature_tensor)  # 0对应<UNK>
            else:
                augmented_features[feature_name] = feature_tensor
        
        return augmented_features


class MultiTaskDataset(Dataset):
    """多任务数据集"""
    
    def __init__(self, features: Dict[str, torch.Tensor], 
                 targets: Dict[str, torch.Tensor],
                 scenario_features: Optional[Dict[str, torch.Tensor]] = None,
                 augmentation: Optional[DataAugmentation] = None):
        
        self.features = features
        self.targets = targets
        self.scenario_features = scenario_features
        self.augmentation = augmentation
        
        # 验证数据长度一致性
        feature_lengths = [len(v) for v in features.values()]
        target_lengths = [len(v) for v in targets.values()]
        
        if not all(l == feature_lengths[0] for l in feature_lengths):
            raise ValueError("All features must have the same length")
        if not all(l == target_lengths[0] for l in target_lengths):
            raise ValueError("All targets must have the same length")
        if feature_lengths[0] != target_lengths[0]:
            raise ValueError("Features and targets must have the same length")
            
        self.length = feature_lengths[0]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 获取特征
        batch_features = {name: tensor[idx] for name, tensor in self.features.items()}
        
        # 获取目标
        batch_targets = {name: tensor[idx] for name, tensor in self.targets.items()}
        
        # 获取场景特征
        batch_scenario_features = None
        if self.scenario_features is not None:
            batch_scenario_features = {name: tensor[idx] for name, tensor in self.scenario_features.items()}
        
        # 应用数据增强（仅在训练时）
        if self.augmentation is not None and self.training:
            batch_features = self.augmentation.feature_dropout(batch_features)
        
        return {
            'features': batch_features,
            'targets': batch_targets,
            'scenario_features': batch_scenario_features
        }


class DataSplitter:
    """数据分割工具"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.val_split = config.get('val_split', 0.2)
        self.test_split = config.get('test_split', 0.1)
        self.random_state = config.get('random_state', 42)
        self.stratify = config.get('stratify', True)
    
    def split_data(self, data: pd.DataFrame, targets: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """分割数据为训练、验证和测试集"""
        
        # 选择用于分层的主要任务（通常是CTR）
        stratify_column = targets.get('ctr', list(targets.values())[0])
        
        if self.stratify and stratify_column in data.columns:
            stratify_array = data[stratify_column]
        else:
            stratify_array = None
        
        # 首先分离出测试集
        if self.test_split > 0:
            train_val_data, test_data = train_test_split(
                data, 
                test_size=self.test_split,
                stratify=stratify_array,
                random_state=self.random_state
            )
        else:
            train_val_data = data
            test_data = pd.DataFrame()
        
        # 分离训练集和验证集
        if self.val_split > 0:
            if self.stratify and stratify_column in train_val_data.columns:
                stratify_array = train_val_data[stratify_column]
            else:
                stratify_array = None
                
            train_data, val_data = train_test_split(
                train_val_data,
                test_size=self.val_split / (1 - self.test_split),  # 调整验证集比例
                stratify=stratify_array,
                random_state=self.random_state
            )
        else:
            train_data = train_val_data
            val_data = pd.DataFrame()
        
        print(f"数据分割完成:")
        print(f"  训练集: {len(train_data)} 样本")
        print(f"  验证集: {len(val_data)} 样本")
        print(f"  测试集: {len(test_data)} 样本")
        
        return train_data, val_data, test_data
    
    def create_cv_splits(self, data: pd.DataFrame, targets: Dict[str, str], 
                        n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """创建交叉验证分割"""
        
        stratify_column = targets.get('ctr', list(targets.values())[0])
        
        if self.stratify and stratify_column in data.columns:
            stratify_array = data[stratify_column]
        else:
            stratify_array = None
        
        if stratify_array is not None:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            splits = list(skf.split(data, stratify_array))
        else:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            splits = list(kf.split(data))
        
        return splits


def create_data_loaders(train_dataset: MultiTaskDataset,
                       val_dataset: Optional[MultiTaskDataset] = None,
                       test_dataset: Optional[MultiTaskDataset] = None,
                       batch_size: int = 2048,
                       num_workers: int = 4,
                       weighted_sampler: Optional[WeightedRandomSampler] = None) -> Dict[str, DataLoader]:
    """创建数据加载器"""
    
    loaders = {}
    
    # 训练加载器
    loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(weighted_sampler is None),
        sampler=weighted_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # 验证加载器
    if val_dataset is not None:
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # 测试加载器
    if test_dataset is not None:
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return loaders


def print_data_statistics(data: pd.DataFrame, targets: Dict[str, str]):
    """打印数据统计信息"""
    
    print(f"\n数据统计信息:")
    print(f"总样本数: {len(data)}")
    print(f"特征数量: {len(data.columns) - len(targets)}")
    
    print(f"\n各任务标签分布:")
    for task_name, column_name in targets.items():
        if column_name in data.columns:
            value_counts = data[column_name].value_counts()
            total = len(data)
            print(f"  {task_name}:")
            for value, count in value_counts.items():
                print(f"    {value}: {count} ({count/total:.2%})")
        else:
            print(f"  {task_name}: 列 '{column_name}' 不存在")
    
    print(f"\n缺失值统计:")
    missing_counts = data.isnull().sum()
    if missing_counts.sum() > 0:
        for column, count in missing_counts.items():
            if count > 0:
                print(f"  {column}: {count} ({count/len(data):.2%})")
    else:
        print("  无缺失值")