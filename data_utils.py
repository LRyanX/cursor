import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import logging

class SampleBalancer:
    """样本均衡器，处理不同类型的样本不均衡问题"""
    
    def __init__(self, strategy: str = 'weighted', random_state: int = 42):
        """
        Args:
            strategy: 均衡策略，可选 'weighted', 'oversample', 'undersample', 'combine'
            random_state: 随机种子
        """
        self.strategy = strategy
        self.random_state = random_state
        self.samplers = {}
        
    def fit_transform(self, 
                     X: np.ndarray, 
                     y_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        对多任务数据进行样本均衡处理
        
        Args:
            X: 特征数据
            y_dict: 各任务的标签字典
            
        Returns:
            均衡后的特征、标签和样本权重
        """
        if self.strategy == 'weighted':
            return self._compute_sample_weights(X, y_dict)
        elif self.strategy == 'oversample':
            return self._oversample(X, y_dict)
        elif self.strategy == 'undersample':
            return self._undersample(X, y_dict)
        elif self.strategy == 'combine':
            return self._combine_sampling(X, y_dict)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _compute_sample_weights(self, 
                               X: np.ndarray, 
                               y_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """计算样本权重"""
        sample_weights = {}
        
        for task, y in y_dict.items():
            # 处理缺失值
            valid_mask = ~np.isnan(y)
            if not np.any(valid_mask):
                sample_weights[task] = np.ones_like(y)
                continue
                
            y_valid = y[valid_mask]
            
            # 计算类别权重
            try:
                unique_classes = np.unique(y_valid)
                if len(unique_classes) > 1:
                    class_weights = class_weight.compute_class_weight(
                        'balanced',
                        classes=unique_classes,
                        y=y_valid
                    )
                    weight_dict = dict(zip(unique_classes, class_weights))
                    
                    # 为每个样本分配权重
                    weights = np.ones_like(y, dtype=float)
                    for cls, weight in weight_dict.items():
                        weights[y == cls] = weight
                    
                    # 对缺失值样本设置较小权重
                    weights[~valid_mask] = 0.1
                    
                    sample_weights[task] = weights
                else:
                    sample_weights[task] = np.ones_like(y)
            except Exception as e:
                logging.warning(f"Failed to compute class weights for task {task}: {e}")
                sample_weights[task] = np.ones_like(y)
        
        return X, y_dict, sample_weights
    
    def _oversample(self, 
                   X: np.ndarray, 
                   y_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """过采样处理"""
        # 选择主要任务进行采样（通常是CTR）
        main_task = list(y_dict.keys())[0]
        y_main = y_dict[main_task]
        
        # 处理缺失值
        valid_mask = ~np.isnan(y_main)
        X_valid = X[valid_mask]
        y_main_valid = y_main[valid_mask]
        
        if len(np.unique(y_main_valid)) < 2:
            return self._compute_sample_weights(X, y_dict)
        
        # 使用SMOTE进行过采样
        try:
            smote = SMOTE(random_state=self.random_state)
            X_resampled, y_main_resampled = smote.fit_resample(X_valid, y_main_valid)
            
            # 为其他任务生成对应的标签
            y_dict_resampled = {main_task: y_main_resampled}
            
            # 对于其他任务，使用最近邻方法进行标签传播
            for task, y_task in y_dict.items():
                if task != main_task:
                    y_task_valid = y_task[valid_mask]
                    # 简单的重复采样，实际应用中可以使用更复杂的方法
                    indices = smote.sample_indices_
                    y_dict_resampled[task] = y_task_valid[indices]
            
            # 计算重采样后的权重
            sample_weights = {}
            for task in y_dict_resampled:
                sample_weights[task] = np.ones(len(y_dict_resampled[task]))
            
            return X_resampled, y_dict_resampled, sample_weights
            
        except Exception as e:
            logging.warning(f"SMOTE failed: {e}, falling back to weighted sampling")
            return self._compute_sample_weights(X, y_dict)
    
    def _undersample(self, 
                    X: np.ndarray, 
                    y_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """欠采样处理"""
        main_task = list(y_dict.keys())[0]
        y_main = y_dict[main_task]
        
        valid_mask = ~np.isnan(y_main)
        X_valid = X[valid_mask]
        y_main_valid = y_main[valid_mask]
        
        if len(np.unique(y_main_valid)) < 2:
            return self._compute_sample_weights(X, y_dict)
        
        try:
            undersampler = RandomUnderSampler(random_state=self.random_state)
            X_resampled, y_main_resampled = undersampler.fit_resample(X_valid, y_main_valid)
            
            # 获取采样索引
            indices = undersampler.sample_indices_
            
            y_dict_resampled = {main_task: y_main_resampled}
            for task, y_task in y_dict.items():
                if task != main_task:
                    y_task_valid = y_task[valid_mask]
                    y_dict_resampled[task] = y_task_valid[indices]
            
            sample_weights = {}
            for task in y_dict_resampled:
                sample_weights[task] = np.ones(len(y_dict_resampled[task]))
            
            return X_resampled, y_dict_resampled, sample_weights
            
        except Exception as e:
            logging.warning(f"Undersampling failed: {e}, falling back to weighted sampling")
            return self._compute_sample_weights(X, y_dict)
    
    def _combine_sampling(self, 
                         X: np.ndarray, 
                         y_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """组合采样策略"""
        main_task = list(y_dict.keys())[0]
        y_main = y_dict[main_task]
        
        valid_mask = ~np.isnan(y_main)
        X_valid = X[valid_mask]
        y_main_valid = y_main[valid_mask]
        
        if len(np.unique(y_main_valid)) < 2:
            return self._compute_sample_weights(X, y_dict)
        
        try:
            # 使用SMOTETomek结合过采样和欠采样
            smote_tomek = SMOTETomek(random_state=self.random_state)
            X_resampled, y_main_resampled = smote_tomek.fit_resample(X_valid, y_main_valid)
            
            # 处理其他任务的标签
            y_dict_resampled = {main_task: y_main_resampled}
            
            # 获取采样索引（需要自定义实现）
            # 这里简化处理，实际应用中需要更复杂的标签对齐
            for task, y_task in y_dict.items():
                if task != main_task:
                    y_task_valid = y_task[valid_mask]
                    # 使用插值或最近邻方法
                    y_dict_resampled[task] = np.repeat(y_task_valid, 
                                                     len(y_main_resampled) // len(y_task_valid) + 1)[:len(y_main_resampled)]
            
            sample_weights = {}
            for task in y_dict_resampled:
                sample_weights[task] = np.ones(len(y_dict_resampled[task]))
            
            return X_resampled, y_dict_resampled, sample_weights
            
        except Exception as e:
            logging.warning(f"Combined sampling failed: {e}, falling back to weighted sampling")
            return self._compute_sample_weights(X, y_dict)

class MultiTaskDataset(data.Dataset):
    """多任务数据集类"""
    
    def __init__(self, 
                 features: np.ndarray,
                 targets: Dict[str, np.ndarray],
                 scene_ids: np.ndarray,
                 sample_weights: Optional[Dict[str, np.ndarray]] = None):
        """
        Args:
            features: 特征数据
            targets: 目标标签字典
            scene_ids: 场景ID
            sample_weights: 样本权重字典
        """
        self.features = torch.FloatTensor(features)
        self.scene_ids = torch.LongTensor(scene_ids)
        
        self.targets = {}
        for task, target in targets.items():
            self.targets[task] = torch.FloatTensor(target)
        
        self.sample_weights = {}
        if sample_weights:
            for task, weights in sample_weights.items():
                self.sample_weights[task] = torch.FloatTensor(weights)
        
        self.length = len(features)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        item = {
            'features': self.features[idx],
            'scene_id': self.scene_ids[idx],
        }
        
        # 添加目标标签
        for task, target in self.targets.items():
            item[f'target_{task}'] = target[idx]
        
        # 添加样本权重
        for task, weights in self.sample_weights.items():
            item[f'weight_{task}'] = weights[idx]
        
        return item

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, 
                 balance_strategy: str = 'weighted',
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42):
        """
        Args:
            balance_strategy: 样本平衡策略
            test_size: 测试集比例
            val_size: 验证集比例
            random_state: 随机种子
        """
        self.balance_strategy = balance_strategy
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.balancer = SampleBalancer(balance_strategy, random_state)
        
    def prepare_data(self, 
                    df: pd.DataFrame,
                    feature_cols: List[str],
                    target_cols: Dict[str, str],
                    scene_col: str) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
        """
        准备训练、验证和测试数据
        
        Args:
            df: 输入数据框
            feature_cols: 特征列名列表
            target_cols: 目标列名字典 {task_name: column_name}
            scene_col: 场景列名
            
        Returns:
            训练、验证、测试数据加载器
        """
        # 提取特征和标签
        X = df[feature_cols].values.astype(np.float32)
        y_dict = {}
        for task, col in target_cols.items():
            y_dict[task] = df[col].values.astype(np.float32)
        
        scene_ids = df[scene_col].values.astype(np.int64)
        
        # 数据分割
        X_temp, X_test, y_temp_dict, y_test_dict, scene_temp, scene_test = self._split_data(
            X, y_dict, scene_ids, test_size=self.test_size
        )
        
        # 进一步分割训练和验证集
        adjusted_val_size = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train_dict, y_val_dict, scene_train, scene_val = self._split_data(
            X_temp, y_temp_dict, scene_temp, test_size=adjusted_val_size
        )
        
        # 对训练集进行样本平衡处理
        X_train_balanced, y_train_balanced, weights_train = self.balancer.fit_transform(
            X_train, y_train_dict
        )
        
        # 创建数据集
        train_dataset = MultiTaskDataset(
            X_train_balanced, y_train_balanced, scene_train, weights_train
        )
        
        val_dataset = MultiTaskDataset(
            X_val, y_val_dict, scene_val
        )
        
        test_dataset = MultiTaskDataset(
            X_test, y_test_dict, scene_test
        )
        
        # 创建数据加载器
        train_loader = data.DataLoader(
            train_dataset, batch_size=256, shuffle=True, num_workers=4
        )
        
        val_loader = data.DataLoader(
            val_dataset, batch_size=512, shuffle=False, num_workers=4
        )
        
        test_loader = data.DataLoader(
            test_dataset, batch_size=512, shuffle=False, num_workers=4
        )
        
        return train_loader, val_loader, test_loader
    
    def _split_data(self, 
                   X: np.ndarray, 
                   y_dict: Dict[str, np.ndarray], 
                   scene_ids: np.ndarray, 
                   test_size: float) -> Tuple:
        """分割数据"""
        # 使用主任务进行分层抽样
        main_task = list(y_dict.keys())[0]
        y_main = y_dict[main_task]
        
        # 处理缺失值，用于分层
        stratify_y = y_main.copy()
        nan_mask = np.isnan(stratify_y)
        if np.any(nan_mask):
            stratify_y[nan_mask] = -1  # 将NaN设为特殊类别
        
        # 确保每个类别至少有2个样本
        unique_vals, counts = np.unique(stratify_y, return_counts=True)
        if np.any(counts < 2):
            stratify_y = None
        
        # 数据分割
        indices = np.arange(len(X))
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            stratify=stratify_y,
            random_state=self.random_state
        )
        
        # 分割特征
        X_train, X_test = X[train_indices], X[test_indices]
        scene_train, scene_test = scene_ids[train_indices], scene_ids[test_indices]
        
        # 分割标签
        y_train_dict, y_test_dict = {}, {}
        for task, y in y_dict.items():
            y_train_dict[task] = y[train_indices]
            y_test_dict[task] = y[test_indices]
        
        return X_train, X_test, y_train_dict, y_test_dict, scene_train, scene_test
    
    def get_data_stats(self, y_dict: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """获取数据统计信息"""
        stats = {}
        
        for task, y in y_dict.items():
            valid_mask = ~np.isnan(y)
            y_valid = y[valid_mask]
            
            if len(y_valid) > 0:
                unique_vals, counts = np.unique(y_valid, return_counts=True)
                stats[task] = {
                    'total_samples': len(y),
                    'valid_samples': len(y_valid),
                    'missing_ratio': 1 - len(y_valid) / len(y),
                    'class_distribution': dict(zip(unique_vals, counts)),
                    'positive_ratio': np.mean(y_valid) if len(unique_vals) == 2 else None
                }
            else:
                stats[task] = {
                    'total_samples': len(y),
                    'valid_samples': 0,
                    'missing_ratio': 1.0,
                    'class_distribution': {},
                    'positive_ratio': None
                }
        
        return stats

def create_sample_data(num_samples: int = 10000, 
                      num_features: int = 100, 
                      num_scenes: int = 5) -> pd.DataFrame:
    """创建示例数据用于测试"""
    np.random.seed(42)
    
    # 生成特征
    features = np.random.randn(num_samples, num_features)
    
    # 生成场景ID
    scene_ids = np.random.randint(0, num_scenes, num_samples)
    
    # 生成CTR标签（点击率）
    ctr_logits = np.sum(features[:, :10], axis=1) + np.random.randn(num_samples) * 0.5
    ctr_probs = 1 / (1 + np.exp(-ctr_logits))
    ctr_labels = (np.random.rand(num_samples) < ctr_probs).astype(float)
    
    # 生成CVR标签（转化率，只在点击的基础上）
    cvr_logits = np.sum(features[:, 10:20], axis=1) + np.random.randn(num_samples) * 0.3
    cvr_probs = 1 / (1 + np.exp(-cvr_logits))
    cvr_labels = np.where(ctr_labels == 1, 
                         (np.random.rand(num_samples) < cvr_probs).astype(float),
                         np.nan)  # 没有点击就没有转化标签
    
    # 生成IVR标签（曝光转化率）
    ivr_logits = np.sum(features[:, 20:30], axis=1) + np.random.randn(num_samples) * 0.4
    ivr_probs = 1 / (1 + np.exp(-ivr_logits))
    ivr_labels = (np.random.rand(num_samples) < ivr_probs * 0.1).astype(float)  # 更稀疏
    
    # 创建DataFrame
    df = pd.DataFrame()
    
    # 添加特征
    for i in range(num_features):
        df[f'feature_{i}'] = features[:, i]
    
    df['scene_id'] = scene_ids
    df['ctr'] = ctr_labels
    df['cvr'] = cvr_labels
    df['ivr'] = ivr_labels
    
    return df