"""
特征处理模块：处理稀疏特征的编码、哈希和词汇表构建
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import pickle
import os
import hashlib
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder
import joblib

class FeatureProcessor:
    """特征处理器，支持稀疏特征编码、哈希编码、词汇表管理"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sparse_features = config.get('sparse_features', [])
        self.scenario_features = config.get('scenario_features', [])
        self.max_vocab_size = config.get('max_vocab_size', 10000)
        self.min_frequency = config.get('min_frequency', 5)
        self.use_hash_encoding = config.get('use_hash_encoding', True)
        self.hash_bucket_size = config.get('hash_bucket_size', 100000)
        
        # 存储词汇表和编码器
        self.vocabularies = {}
        self.encoders = {}
        self.feature_dims = {}
        self.value_counts = {}
        self.is_fitted = False
        
    def _hash_feature(self, feature_name: str, value: str, bucket_size: int) -> int:
        """使用MD5哈希将特征值映射到固定大小的桶中"""
        hash_input = f"{feature_name}_{value}".encode('utf-8')
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        return hash_value % bucket_size
    
    def _build_vocabulary(self, series: pd.Series, feature_name: str) -> Dict:
        """为单个特征构建词汇表"""
        # 统计频次
        value_counts = series.value_counts()
        self.value_counts[feature_name] = value_counts
        
        # 过滤低频词
        filtered_values = value_counts[value_counts >= self.min_frequency]
        
        # 限制词汇表大小
        if len(filtered_values) > self.max_vocab_size:
            filtered_values = filtered_values.head(self.max_vocab_size)
        
        # 构建词汇表：value -> index
        vocab = {'<UNK>': 0, '<PAD>': 1}  # 预留特殊token
        for i, value in enumerate(filtered_values.index, start=2):
            vocab[str(value)] = i
            
        return vocab
    
    def fit(self, data: pd.DataFrame) -> 'FeatureProcessor':
        """拟合特征处理器"""
        print("开始构建特征词汇表...")
        
        for feature in self.sparse_features:
            if feature not in data.columns:
                print(f"警告: 特征 {feature} 不在数据中")
                continue
                
            print(f"处理特征: {feature}")
            
            # 处理缺失值
            series = data[feature].fillna('<UNK>').astype(str)
            
            if self.use_hash_encoding and len(series.unique()) > self.max_vocab_size:
                # 使用哈希编码处理高基数特征
                print(f"  特征 {feature} 使用哈希编码 (唯一值数量: {len(series.unique())})")
                self.vocabularies[feature] = 'hash'
                self.feature_dims[feature] = self.hash_bucket_size
            else:
                # 构建词汇表
                vocab = self._build_vocabulary(series, feature)
                self.vocabularies[feature] = vocab
                self.feature_dims[feature] = len(vocab)
                print(f"  特征 {feature} 词汇表大小: {len(vocab)}")
        
        self.is_fitted = True
        print("特征词汇表构建完成")
        return self
    
    def transform(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """转换数据为张量格式"""
        if not self.is_fitted:
            raise ValueError("FeatureProcessor must be fitted before transform")
        
        result = {}
        
        for feature in self.sparse_features:
            if feature not in data.columns:
                continue
                
            series = data[feature].fillna('<UNK>').astype(str)
            
            if self.vocabularies[feature] == 'hash':
                # 哈希编码
                indices = [self._hash_feature(feature, str(val), self.hash_bucket_size) 
                          for val in series]
            else:
                # 词汇表编码
                vocab = self.vocabularies[feature]
                indices = [vocab.get(str(val), vocab['<UNK>']) for val in series]
            
            result[feature] = torch.tensor(indices, dtype=torch.long)
        
        return result
    
    def fit_transform(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """拟合并转换数据"""
        return self.fit(data).transform(data)
    
    def get_scenario_features(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """提取场景特征张量"""
        scenario_tensors = []
        for feature in self.scenario_features:
            if feature in data:
                scenario_tensors.append(data[feature].unsqueeze(1))
        
        if scenario_tensors:
            return torch.cat(scenario_tensors, dim=1)
        else:
            # 如果没有场景特征，返回零向量
            batch_size = next(iter(data.values())).size(0)
            return torch.zeros(batch_size, 1, dtype=torch.long)
    
    def save(self, save_dir: str):
        """保存特征处理器"""
        os.makedirs(save_dir, exist_ok=True)
        
        save_data = {
            'config': self.config,
            'vocabularies': self.vocabularies,
            'feature_dims': self.feature_dims,
            'value_counts': self.value_counts,
            'is_fitted': self.is_fitted
        }
        
        with open(os.path.join(save_dir, 'feature_processor.pkl'), 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"特征处理器已保存到 {save_dir}")
    
    @classmethod
    def load(cls, save_dir: str) -> 'FeatureProcessor':
        """加载特征处理器"""
        with open(os.path.join(save_dir, 'feature_processor.pkl'), 'rb') as f:
            save_data = pickle.load(f)
        
        processor = cls(save_data['config'])
        processor.vocabularies = save_data['vocabularies']
        processor.feature_dims = save_data['feature_dims']
        processor.value_counts = save_data['value_counts']
        processor.is_fitted = save_data['is_fitted']
        
        print(f"特征处理器已从 {save_dir} 加载")
        return processor
    
    def get_feature_info(self) -> Dict:
        """获取特征信息"""
        return {
            'feature_dims': self.feature_dims,
            'total_features': len(self.sparse_features),
            'scenario_features': self.scenario_features,
            'total_vocabulary_size': sum(self.feature_dims.values())
        }


class EmbeddingLayer(nn.Module):
    """嵌入层，支持多个稀疏特征的嵌入"""
    
    def __init__(self, feature_dims: Dict[str, int], embedding_dim: int, 
                 dropout_rate: float = 0.1, l2_reg: float = 1e-6):
        super().__init__()
        self.feature_dims = feature_dims
        self.embedding_dim = embedding_dim
        self.l2_reg = l2_reg
        
        # 创建嵌入层
        self.embeddings = nn.ModuleDict()
        for feature_name, vocab_size in feature_dims.items():
            self.embeddings[feature_name] = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=1
            )
            # 初始化权重
            nn.init.xavier_uniform_(self.embeddings[feature_name].weight)
            # padding位置设为0
            with torch.no_grad():
                self.embeddings[feature_name].weight[1].fill_(0)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        embeddings = []
        
        for feature_name, indices in features.items():
            if feature_name in self.embeddings:
                emb = self.embeddings[feature_name](indices)
                embeddings.append(emb)
        
        if not embeddings:
            raise ValueError("No valid features found in input")
        
        # 拼接所有特征的嵌入向量
        combined_embeddings = torch.cat(embeddings, dim=1)
        return self.dropout(combined_embeddings)
    
    def get_l2_loss(self) -> torch.Tensor:
        """计算嵌入层的L2正则化损失"""
        l2_loss = 0
        for embedding in self.embeddings.values():
            l2_loss += torch.norm(embedding.weight, p=2) ** 2
        return self.l2_reg * l2_loss


class ScenarioEmbedding(nn.Module):
    """场景嵌入层，处理场景相关特征"""
    
    def __init__(self, scenario_feature_dims: Dict[str, int], 
                 scenario_embedding_dim: int, output_dim: int):
        super().__init__()
        self.scenario_feature_dims = scenario_feature_dims
        self.scenario_embedding_dim = scenario_embedding_dim
        
        # 场景特征嵌入
        self.scenario_embeddings = nn.ModuleDict()
        for feature_name, vocab_size in scenario_feature_dims.items():
            self.scenario_embeddings[feature_name] = nn.Embedding(
                vocab_size, scenario_embedding_dim
            )
        
        # 场景融合层
        total_scenario_dim = len(scenario_feature_dims) * scenario_embedding_dim
        self.scenario_fusion = nn.Sequential(
            nn.Linear(total_scenario_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, scenario_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        scenario_embs = []
        
        for feature_name, indices in scenario_features.items():
            if feature_name in self.scenario_embeddings:
                emb = self.scenario_embeddings[feature_name](indices)
                scenario_embs.append(emb)
        
        if scenario_embs:
            scenario_concat = torch.cat(scenario_embs, dim=1)
            return self.scenario_fusion(scenario_concat)
        else:
            # 如果没有场景特征，返回零向量
            batch_size = next(iter(scenario_features.values())).size(0)
            return torch.zeros(batch_size, self.scenario_fusion[0].out_features, 
                             device=next(self.parameters()).device)