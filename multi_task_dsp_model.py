import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import math

class SceneAdaptationLayer(nn.Module):
    """场景适配层，为不同场景学习特定的特征变换"""
    
    def __init__(self, input_dim: int, num_scenes: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
            
        self.num_scenes = num_scenes
        self.input_dim = input_dim
        
        # 为每个场景创建独立的变换层
        self.scene_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(num_scenes)
        ])
        
        # 场景注意力机制
        self.scene_attention = nn.Sequential(
            nn.Linear(input_dim, num_scenes),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor, scene_ids: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 计算场景注意力权重
        scene_weights = self.scene_attention(x)  # [batch_size, num_scenes]
        
        # 对每个场景进行变换
        scene_outputs = []
        for i in range(self.num_scenes):
            transformed = self.scene_transforms[i](x)
            scene_outputs.append(transformed)
        
        scene_outputs = torch.stack(scene_outputs, dim=1)  # [batch_size, num_scenes, input_dim]
        
        # 使用注意力权重进行加权融合
        output = torch.sum(scene_weights.unsqueeze(-1) * scene_outputs, dim=1)
        
        return output

class MultiTaskTower(nn.Module):
    """多任务塔网络，为每个任务学习特定的表示"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], task_name: str):
        super().__init__()
        self.task_name = task_name
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # 最后一层输出
        layers.append(nn.Linear(prev_dim, 1))
        
        self.tower = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tower(x)

class MMOEExpertNet(nn.Module):
    """MMoE专家网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        return torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, hidden_dim]

class GatingNetwork(nn.Module):
    """门控网络，为每个任务分配专家权重"""
    
    def __init__(self, input_dim: int, num_experts: int, num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=-1)
            ) for _ in range(num_tasks)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_outputs = []
        for gate in self.gates:
            gate_outputs.append(gate(x))
        return torch.stack(gate_outputs, dim=1)  # [batch_size, num_tasks, num_experts]

class CalibrationLayer(nn.Module):
    """校准层，用于对齐预估值和真实值"""
    
    def __init__(self, num_bins: int = 10):
        super().__init__()
        self.num_bins = num_bins
        # 使用可学习的校准参数
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        # 使用Platt Scaling进行校准
        calibrated = torch.sigmoid(self.alpha * predictions + self.beta)
        return calibrated

class FocalLoss(nn.Module):
    """Focal Loss，解决样本不均衡问题"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class GradientBalancer:
    """梯度平衡器，解决跷跷板问题"""
    
    def __init__(self, num_tasks: int, alpha: float = 0.5):
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.task_weights = [1.0] * num_tasks
        self.prev_losses = [0.0] * num_tasks
        
    def update_weights(self, current_losses: List[float]):
        """动态更新任务权重"""
        for i in range(self.num_tasks):
            if self.prev_losses[i] > 0:
                loss_ratio = current_losses[i] / self.prev_losses[i]
                # 如果损失下降缓慢，增加权重
                if loss_ratio > 0.95:
                    self.task_weights[i] *= (1 + self.alpha)
                # 如果损失下降很快，减少权重
                elif loss_ratio < 0.8:
                    self.task_weights[i] *= (1 - self.alpha)
        
        # 归一化权重
        total_weight = sum(self.task_weights)
        self.task_weights = [w / total_weight * self.num_tasks for w in self.task_weights]
        self.prev_losses = current_losses.copy()
        
        return self.task_weights

class MultiTaskMultiSceneDSPModel(nn.Module):
    """多任务多场景DSP广告模型"""
    
    def __init__(self, 
                 feature_dim: int,
                 num_scenes: int = 5,
                 num_experts: int = 6,
                 expert_hidden_dim: int = 256,
                 tower_hidden_dims: List[int] = [256, 128],
                 tasks: List[str] = ['ctr', 'cvr', 'ivr']):
        super().__init__()
        
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.num_scenes = num_scenes
        
        # 特征嵌入层
        self.feature_embedding = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 场景适配层
        self.scene_adapter = SceneAdaptationLayer(512, num_scenes)
        
        # MMoE专家网络
        self.expert_net = MMOEExpertNet(512, expert_hidden_dim, num_experts)
        
        # 门控网络
        self.gating_net = GatingNetwork(512, num_experts, self.num_tasks)
        
        # 任务特定塔网络
        self.task_towers = nn.ModuleDict({
            task: MultiTaskTower(expert_hidden_dim, tower_hidden_dims, task)
            for task in tasks
        })
        
        # 校准层
        self.calibration_layers = nn.ModuleDict({
            task: CalibrationLayer() for task in tasks
        })
        
        # 损失函数
        self.focal_losses = nn.ModuleDict({
            task: FocalLoss(alpha=1.0, gamma=2.0) for task in tasks
        })
        
        # 梯度平衡器
        self.gradient_balancer = GradientBalancer(self.num_tasks)
        
    def forward(self, 
                features: torch.Tensor, 
                scene_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # 特征嵌入
        embedded_features = self.feature_embedding(features)
        
        # 场景适配
        scene_adapted_features = self.scene_adapter(embedded_features, scene_ids)
        
        # 专家网络输出
        expert_outputs = self.expert_net(scene_adapted_features)  # [batch_size, num_experts, hidden_dim]
        
        # 门控网络输出
        gate_weights = self.gating_net(scene_adapted_features)  # [batch_size, num_tasks, num_experts]
        
        # 为每个任务计算加权专家输出
        task_outputs = {}
        for i, task in enumerate(self.tasks):
            # 获取当前任务的门控权重
            task_gate = gate_weights[:, i, :].unsqueeze(-1)  # [batch_size, num_experts, 1]
            
            # 加权求和专家输出
            weighted_expert_output = torch.sum(task_gate * expert_outputs, dim=1)  # [batch_size, hidden_dim]
            
            # 通过任务特定塔网络
            tower_output = self.task_towers[task](weighted_expert_output)
            
            # 校准
            calibrated_output = self.calibration_layers[task](tower_output.squeeze(-1))
            
            task_outputs[task] = calibrated_output
        
        return task_outputs
    
    def compute_loss(self, 
                     predictions: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor],
                     sample_weights: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        task_losses = {}
        total_loss = 0
        
        # 计算每个任务的损失
        for task in self.tasks:
            if task in targets and targets[task] is not None:
                pred = predictions[task]
                target = targets[task].float()
                
                # 使用Focal Loss
                loss = self.focal_losses[task](pred, target)
                
                # 如果提供了样本权重，应用权重
                if sample_weights and task in sample_weights:
                    loss = loss * sample_weights[task]
                    loss = loss.mean()
                
                task_losses[task] = loss.item()
                total_loss += loss
        
        # 动态调整任务权重
        if len(task_losses) > 1:
            weights = self.gradient_balancer.update_weights(list(task_losses.values()))
            weighted_loss = 0
            for i, task in enumerate(self.tasks):
                if task in task_losses:
                    weighted_loss += weights[i] * task_losses[task]
            total_loss = weighted_loss
        
        return total_loss, task_losses
    
    def predict(self, features: torch.Tensor, scene_ids: torch.Tensor) -> Dict[str, np.ndarray]:
        """预测接口"""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(features, scene_ids)
            
        # 转换为numpy数组
        results = {}
        for task, pred in predictions.items():
            results[task] = pred.cpu().numpy()
        
        return results