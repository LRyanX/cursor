"""
多任务多场景模型核心架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

from .feature_processor import EmbeddingLayer, ScenarioEmbedding


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(context)
        return output


class CrossStitchUnit(nn.Module):
    """Cross-stitch单元，用于任务间知识共享"""
    
    def __init__(self, num_tasks: int, hidden_dim: int):
        super().__init__()
        self.num_tasks = num_tasks
        self.hidden_dim = hidden_dim
        
        # 学习任务间的线性组合权重
        self.cross_stitch = nn.Parameter(torch.eye(num_tasks))
        
    def forward(self, task_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        task_outputs: List of tensors, each of shape [batch_size, hidden_dim]
        """
        if len(task_outputs) != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} task outputs, got {len(task_outputs)}")
        
        # Stack task outputs
        stacked = torch.stack(task_outputs, dim=0)  # [num_tasks, batch_size, hidden_dim]
        
        # Apply cross-stitch transformation
        cross_stitched = torch.einsum('ij,jbh->ibh', self.cross_stitch, stacked)
        
        # Return list of transformed outputs
        return [cross_stitched[i] for i in range(self.num_tasks)]


class ScenarioGate(nn.Module):
    """场景门控机制，根据场景调整特征权重"""
    
    def __init__(self, input_dim: int, scenario_dim: int, gate_dim: int = 64):
        super().__init__()
        
        self.gate_network = nn.Sequential(
            nn.Linear(scenario_dim, gate_dim),
            nn.ReLU(),
            nn.Linear(gate_dim, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor, scenario_embedding: torch.Tensor) -> torch.Tensor:
        """
        features: [batch_size, input_dim]
        scenario_embedding: [batch_size, scenario_dim]
        """
        gate_weights = self.gate_network(scenario_embedding)
        return features * gate_weights


class TaskSpecificTower(nn.Module):
    """任务特定塔网络"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 dropout_rate: float = 0.3, use_batch_norm: bool = True):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.tower = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tower(x)


class UncertaintyWeighting(nn.Module):
    """基于不确定性的多任务权重学习"""
    
    def __init__(self, num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks
        # 学习每个任务的不确定性参数
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        losses: List of task losses
        Returns: Weighted total loss
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        
        return total_loss


class MultiTaskMultiScenarioModel(nn.Module):
    """多任务多场景广告预估模型"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.num_tasks = len(config['tasks'])
        self.tasks = config['tasks']
        
        # 模型配置
        model_config = config['model_config']
        self.embedding_dim = model_config['embedding_dim']
        self.hidden_dims = model_config['hidden_dims']
        self.dropout_rate = model_config['dropout_rate']
        self.use_batch_norm = model_config['batch_norm']
        self.scenario_embedding_dim = model_config['scenario_embedding_dim']
        self.tower_hidden_dims = model_config['tower_hidden_dims']
        
        # 特征维度（需要在训练时设置）
        self.feature_dims = None
        self.scenario_feature_dims = None
        
    def build_model(self, feature_dims: Dict[str, int], scenario_feature_dims: Dict[str, int]):
        """构建模型层，在知道特征维度后调用"""
        self.feature_dims = feature_dims
        self.scenario_feature_dims = scenario_feature_dims
        
        # 1. 嵌入层
        self.embedding_layer = EmbeddingLayer(
            feature_dims=feature_dims,
            embedding_dim=self.embedding_dim,
            dropout_rate=self.dropout_rate,
            l2_reg=self.config['regularization_config']['embedding_l2']
        )
        
        # 2. 场景嵌入层
        if scenario_feature_dims:
            self.scenario_embedding = ScenarioEmbedding(
                scenario_feature_dims=scenario_feature_dims,
                scenario_embedding_dim=self.scenario_embedding_dim,
                output_dim=self.scenario_embedding_dim
            )
        else:
            self.scenario_embedding = None
        
        # 计算输入维度
        total_embedding_dim = len(feature_dims) * self.embedding_dim
        
        # 3. 共享底层网络
        shared_layers = []
        prev_dim = total_embedding_dim
        
        for hidden_dim in self.hidden_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.use_batch_norm:
                shared_layers.append(nn.BatchNorm1d(hidden_dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*shared_layers)
        shared_output_dim = prev_dim
        
        # 4. 场景门控机制
        if self.scenario_embedding is not None:
            self.scenario_gates = nn.ModuleList([
                ScenarioGate(
                    input_dim=shared_output_dim,
                    scenario_dim=self.scenario_embedding_dim
                ) for _ in range(self.num_tasks)
            ])
        else:
            self.scenario_gates = None
        
        # 5. 多头注意力（用于特征交互）
        self.attention = MultiHeadAttention(
            d_model=shared_output_dim,
            n_heads=8,
            dropout=self.dropout_rate
        )
        
        # 6. Cross-stitch单元（任务间知识共享）
        self.cross_stitch = CrossStitchUnit(
            num_tasks=self.num_tasks,
            hidden_dim=shared_output_dim
        )
        
        # 7. 任务特定塔
        self.task_towers = nn.ModuleList([
            TaskSpecificTower(
                input_dim=shared_output_dim,
                hidden_dims=self.tower_hidden_dims,
                dropout_rate=self.dropout_rate,
                use_batch_norm=self.use_batch_norm
            ) for _ in range(self.num_tasks)
        ])
        
        # 8. 不确定性权重学习
        if self.config['multitask_config']['use_uncertainty_weighting']:
            self.uncertainty_weighting = UncertaintyWeighting(self.num_tasks)
        else:
            self.uncertainty_weighting = None
        
        # 9. 校准层
        if self.config['calibration_config']['temperature_scaling']:
            self.temperature_parameters = nn.Parameter(torch.ones(self.num_tasks))
        else:
            self.temperature_parameters = None
    
    def forward(self, features: Dict[str, torch.Tensor], 
                scenario_features: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """前向传播"""
        
        # 1. 特征嵌入
        embedded_features = self.embedding_layer(features)  # [batch_size, total_embedding_dim]
        
        # 2. 共享网络
        shared_output = self.shared_network(embedded_features)  # [batch_size, shared_output_dim]
        
        # 3. 注意力机制（自注意力）
        # 为注意力机制重塑输入
        batch_size = shared_output.size(0)
        attention_input = shared_output.unsqueeze(1)  # [batch_size, 1, shared_output_dim]
        attended_output = self.attention(attention_input, attention_input, attention_input)
        attended_output = attended_output.squeeze(1)  # [batch_size, shared_output_dim]
        
        # 4. 场景感知处理
        if self.scenario_embedding is not None and scenario_features is not None:
            scenario_emb = self.scenario_embedding(scenario_features)  # [batch_size, scenario_dim]
            
            # 为每个任务应用场景门控
            task_inputs = []
            for i in range(self.num_tasks):
                gated_features = self.scenario_gates[i](attended_output, scenario_emb)
                task_inputs.append(gated_features)
        else:
            # 如果没有场景特征，所有任务使用相同输入
            task_inputs = [attended_output] * self.num_tasks
        
        # 5. Cross-stitch知识共享
        task_inputs = self.cross_stitch(task_inputs)
        
        # 6. 任务特定预测
        task_outputs = {}
        raw_outputs = []
        
        for i, task_name in enumerate(self.tasks):
            task_output = self.task_towers[i](task_inputs[i])  # [batch_size, 1]
            raw_outputs.append(task_output)
            
            # 应用温度缩放校准
            if self.temperature_parameters is not None:
                task_output = task_output / self.temperature_parameters[i]
            
            # Sigmoid激活
            task_outputs[task_name] = torch.sigmoid(task_output.squeeze(-1))  # [batch_size]
        
        # 存储原始输出用于校准
        task_outputs['raw_outputs'] = raw_outputs
        
        return task_outputs
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor],
                     loss_config: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算损失"""
        
        task_losses = {}
        total_loss = 0
        
        # 1. 计算各任务损失
        for i, task_name in enumerate(self.tasks):
            pred = predictions[task_name]
            target = targets[task_name]
            
            if loss_config['use_focal_loss']:
                # Focal Loss处理样本不均衡
                alpha = loss_config['focal_alpha'][i]
                gamma = loss_config['focal_gamma']
                
                ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
                pt = torch.where(target == 1, pred, 1 - pred)
                focal_weight = alpha * (1 - pt) ** gamma
                focal_loss = focal_weight * ce_loss
                task_loss = focal_loss.mean()
            else:
                # 标准BCE损失
                if loss_config['use_class_weights']:
                    # 计算类别权重
                    pos_weight = (target == 0).sum().float() / (target == 1).sum().float()
                    task_loss = F.binary_cross_entropy(pred, target, 
                                                     weight=torch.where(target == 1, pos_weight, 1.0))
                else:
                    task_loss = F.binary_cross_entropy(pred, target)
            
            # 标签平滑
            if self.config['regularization_config']['label_smoothing'] > 0:
                smooth_factor = self.config['regularization_config']['label_smoothing']
                smoothed_target = target * (1 - smooth_factor) + 0.5 * smooth_factor
                task_loss = F.binary_cross_entropy(pred, smoothed_target)
            
            task_losses[task_name] = task_loss
        
        # 2. 合并多任务损失
        if self.uncertainty_weighting is not None:
            # 使用不确定性权重
            loss_list = [task_losses[task] for task in self.tasks]
            total_loss = self.uncertainty_weighting(loss_list)
        else:
            # 使用固定权重
            task_weights = self.config['multitask_config']['task_weights']
            for i, task_name in enumerate(self.tasks):
                total_loss += task_weights[i] * task_losses[task_name]
        
        # 3. 添加正则化损失
        reg_loss = 0
        
        # L2正则化
        if self.config['regularization_config']['l2_lambda'] > 0:
            l2_reg = 0
            for param in self.parameters():
                l2_reg += torch.norm(param, p=2) ** 2
            reg_loss += self.config['regularization_config']['l2_lambda'] * l2_reg
        
        # Embedding L2正则化
        if hasattr(self.embedding_layer, 'get_l2_loss'):
            reg_loss += self.embedding_layer.get_l2_loss()
        
        total_loss += reg_loss
        task_losses['total_loss'] = total_loss
        task_losses['reg_loss'] = reg_loss
        
        return total_loss, task_losses
    
    def get_embeddings(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """获取特征嵌入用于分析"""
        return self.embedding_layer(features)
    
    def get_task_representations(self, features: Dict[str, torch.Tensor],
                                scenario_features: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """获取各任务的中间表示"""
        
        embedded_features = self.embedding_layer(features)
        shared_output = self.shared_network(embedded_features)
        
        attention_input = shared_output.unsqueeze(1)
        attended_output = self.attention(attention_input, attention_input, attention_input)
        attended_output = attended_output.squeeze(1)
        
        task_representations = {}
        
        if self.scenario_embedding is not None and scenario_features is not None:
            scenario_emb = self.scenario_embedding(scenario_features)
            for i, task_name in enumerate(self.tasks):
                gated_features = self.scenario_gates[i](attended_output, scenario_emb)
                task_representations[task_name] = gated_features
        else:
            for task_name in self.tasks:
                task_representations[task_name] = attended_output
        
        return task_representations