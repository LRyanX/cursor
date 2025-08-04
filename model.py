"""
多任务学习模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, Linear
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ScenarioAdapter(nn.Module):
    """场景适配器"""
    def __init__(self, scenario_dim: int, hidden_dim: int, num_scenarios: int = 10):
        super(ScenarioAdapter, self).__init__()
        self.scenario_dim = scenario_dim
        self.hidden_dim = hidden_dim
        self.num_scenarios = num_scenarios
        
        # 场景特定的适配层
        self.scenario_embeddings = nn.Embedding(num_scenarios, scenario_dim)
        self.scenario_projector = nn.Sequential(
            nn.Linear(scenario_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 场景分类器
        self.scenario_classifier = nn.Sequential(
            nn.Linear(scenario_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_scenarios)
        )
    
    def forward(self, scenario_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 场景分类
        scenario_logits = self.scenario_classifier(scenario_features)
        scenario_probs = F.softmax(scenario_logits, dim=1)
        
        # 场景嵌入
        scenario_emb = self.scenario_embeddings.weight  # [num_scenarios, scenario_dim]
        
        # 加权场景嵌入
        weighted_scenario_emb = torch.matmul(scenario_probs, scenario_emb)
        
        # 场景适配
        adapted_features = self.scenario_projector(weighted_scenario_emb)
        
        return adapted_features, scenario_probs

class TaskSpecificHead(nn.Module):
    """任务特定头部"""
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float = 0.3):
        super(TaskSpecificHead, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class MultiTaskModel(nn.Module):
    """多任务学习模型"""
    def __init__(self, model_config, feature_config):
        super(MultiTaskModel, self).__init__()
        self.model_config = model_config
        self.feature_config = feature_config
        
        # 特征维度
        self.feature_dim = len(feature_config.sparse_features) + len(feature_config.numeric_features)
        self.scenario_dim = len(feature_config.scenario_features) * model_config.scenario_embedding_dim
        
        # 共享特征提取器
        self.shared_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, model_config.hidden_dims[0]),
            nn.BatchNorm1d(model_config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(model_config.dropout_rate),
            nn.Linear(model_config.hidden_dims[0], model_config.hidden_dims[1]),
            nn.BatchNorm1d(model_config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(model_config.dropout_rate)
        )
        
        # 场景适配器
        self.scenario_adapter = ScenarioAdapter(
            scenario_dim=model_config.scenario_embedding_dim,
            hidden_dim=model_config.hidden_dims[1],
            num_scenarios=20  # 可调整
        )
        
        # 任务特定头部
        self.task_heads = nn.ModuleDict({
            task: TaskSpecificHead(
                input_dim=model_config.hidden_dims[1],
                hidden_dims=[model_config.hidden_dims[1] // 2],
                dropout_rate=model_config.dropout_rate
            ) for task in model_config.tasks
        })
        
        # 注意力机制（用于任务间交互）
        self.task_attention = nn.MultiheadAttention(
            embed_dim=model_config.hidden_dims[1],
            num_heads=4,
            dropout=model_config.dropout_rate,
            batch_first=True
        )
        
        # 任务权重学习
        self.task_weight_learner = nn.Parameter(torch.ones(len(model_config.tasks)))
        
        # 正则化
        self.l2_reg = model_config.weight_decay
    
    def forward(self, features: torch.Tensor, scenario_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 共享特征提取
        shared_features = self.shared_encoder(features)
        
        # 场景适配
        adapted_features, scenario_probs = self.scenario_adapter(scenario_features)
        
        # 特征融合
        combined_features = shared_features + adapted_features
        
        # 任务注意力
        task_features = combined_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        attended_features, _ = self.task_attention(task_features, task_features, task_features)
        attended_features = attended_features.squeeze(1)
        
        # 任务特定预测
        predictions = {}
        for task in self.model_config.tasks:
            predictions[task] = self.task_heads[task](attended_features)
        
        return predictions
    
    def get_task_weights(self) -> torch.Tensor:
        """获取任务权重"""
        return F.softmax(self.task_weight_learner, dim=0)
    
    def regularization_loss(self) -> torch.Tensor:
        """正则化损失"""
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return self.l2_reg * l2_loss

class ParetoMultiTaskLoss(nn.Module):
    """Pareto多任务损失函数"""
    def __init__(self, tasks: List[str], task_weights: Dict[str, float] = None):
        super(ParetoMultiTaskLoss, self).__init__()
        self.tasks = tasks
        self.task_weights = task_weights or {task: 1.0 for task in tasks}
        
        # 每个任务的Focal Loss
        from utils import FocalLoss
        self.focal_losses = nn.ModuleDict({
            task: FocalLoss(alpha=0.25, gamma=2.0) for task in tasks
        })
        
        # 任务权重学习
        self.log_vars = nn.Parameter(torch.zeros(len(tasks)))
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = {}
        precision_losses = {}
        
        for i, task in enumerate(self.tasks):
            if task in predictions and task in targets:
                # Focal Loss
                focal_loss = self.focal_losses[task](predictions[task], targets[task])
                
                # 精度损失（用于Pareto优化）
                precision = torch.exp(-self.log_vars[i])
                precision_loss = precision * focal_loss + self.log_vars[i]
                
                losses[task] = focal_loss
                precision_losses[task] = precision_loss
        
        # Pareto最优组合
        total_loss = sum(precision_losses.values())
        
        return total_loss, losses

class CalibratedModel(nn.Module):
    """校准模型"""
    def __init__(self, base_model: MultiTaskModel, calibration_method: str = "isotonic"):
        super(CalibratedModel, self).__init__()
        self.base_model = base_model
        self.calibration_method = calibration_method
        self.calibrators = {}
        
    def forward(self, features: torch.Tensor, scenario_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 基础模型预测
        raw_predictions = self.base_model(features, scenario_features)
        
        # 应用校准
        calibrated_predictions = {}
        for task, pred in raw_predictions.items():
            if task in self.calibrators:
                # 转换为概率
                prob_pred = torch.sigmoid(pred)
                # 校准（这里简化处理，实际应该在训练后单独校准）
                calibrated_predictions[task] = prob_pred
            else:
                calibrated_predictions[task] = torch.sigmoid(pred)
        
        return calibrated_predictions
    
    def fit_calibrators(self, val_predictions: Dict[str, np.ndarray], 
                       val_targets: Dict[str, np.ndarray]):
        """训练校准器"""
        from utils import Calibrator
        
        for task in val_predictions.keys():
            calibrator = Calibrator(method=self.calibration_method)
            calibrator.fit(val_predictions[task], val_targets[task])
            self.calibrators[task] = calibrator

class EnsembleModel(nn.Module):
    """集成模型"""
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights or [1.0 / len(models)] * len(models)
    
    def forward(self, features: torch.Tensor, scenario_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        predictions = []
        
        for model in self.models:
            pred = model(features, scenario_features)
            predictions.append(pred)
        
        # 加权平均
        ensemble_pred = {}
        for task in predictions[0].keys():
            task_preds = [pred[task] for pred in predictions]
            weighted_pred = sum(w * p for w, p in zip(self.weights, task_preds))
            ensemble_pred[task] = weighted_pred
        
        return ensemble_pred

def create_model(model_config, feature_config) -> MultiTaskModel:
    """创建模型"""
    model = MultiTaskModel(model_config, feature_config)
    return model

def create_ensemble_models(model_config, feature_config, num_models: int = 3) -> List[MultiTaskModel]:
    """创建集成模型"""
    models = []
    for i in range(num_models):
        # 为每个模型设置不同的随机种子
        torch.manual_seed(model_config.seed + i)
        model = create_model(model_config, feature_config)
        models.append(model)
    return models

class ModelTrainer:
    """模型训练器"""
    def __init__(self, model: MultiTaskModel, model_config, training_config):
        self.model = model
        self.model_config = model_config
        self.training_config = training_config
        self.device = torch.device(model_config.device)
        
        # 损失函数
        self.criterion = ParetoMultiTaskLoss(
            tasks=model_config.tasks,
            task_weights=model_config.task_weights
        )
        
        # 优化器
        if training_config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=model_config.learning_rate,
                weight_decay=training_config.l2_reg
            )
        elif training_config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                lr=model_config.learning_rate,
                momentum=0.9,
                weight_decay=training_config.l2_reg
            )
        
        # 学习率调度器
        if training_config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=model_config.num_epochs
            )
        elif training_config.scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        else:
            self.scheduler = None
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if training_config.use_amp else None
        
        # 训练历史
        self.train_losses = {task: [] for task in model_config.tasks}
        self.val_losses = {task: [] for task in model_config.tasks}
        self.val_metrics = {task: [] for task in model_config.tasks}
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = {task: 0.0 for task in self.model_config.tasks}
        num_batches = 0
        
        for batch in dataloader:
            features = batch['features'].to(self.device)
            targets = {task: target.to(self.device) for task, target in batch['targets'].items()}
            
            # 前向传播
            with torch.cuda.amp.autocast() if self.training_config.use_amp else torch.no_grad():
                predictions = self.model(features, features)  # 简化处理，实际需要分离场景特征
            
            # 计算损失
            total_loss, task_losses = self.criterion(predictions, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            if self.training_config.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()
            
            # 记录损失
            for task, loss in task_losses.items():
                epoch_losses[task] += loss.item()
            num_batches += 1
        
        # 平均损失
        for task in epoch_losses:
            epoch_losses[task] /= num_batches
        
        return epoch_losses
    
    def validate(self, dataloader) -> Tuple[Dict[str, float], Dict[str, float]]:
        """验证"""
        self.model.eval()
        val_losses = {task: 0.0 for task in self.model_config.tasks}
        all_predictions = {task: [] for task in self.model_config.tasks}
        all_targets = {task: [] for task in self.model_config.tasks}
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                targets = {task: target.to(self.device) for task, target in batch['targets'].items()}
                
                predictions = self.model(features, features)
                
                # 计算损失
                total_loss, task_losses = self.criterion(predictions, targets)
                
                # 记录损失和预测
                for task, loss in task_losses.items():
                    val_losses[task] += loss.item()
                    all_predictions[task].extend(torch.sigmoid(predictions[task]).cpu().numpy())
                    all_targets[task].extend(targets[task].cpu().numpy())
                
                num_batches += 1
        
        # 平均损失
        for task in val_losses:
            val_losses[task] /= num_batches
        
        # 计算指标
        from utils import calculate_metrics
        val_metrics = {}
        for task in self.model_config.tasks:
            task_metrics = calculate_metrics(
                np.array(all_targets[task]),
                np.array(all_predictions[task]),
                task
            )
            val_metrics.update(task_metrics)
        
        return val_losses, val_metrics
    
    def train(self, train_dataloader, val_dataloader, save_path: str = None):
        """训练模型"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.model_config.num_epochs):
            # 训练
            train_losses = self.train_epoch(train_dataloader)
            
            # 验证
            val_losses, val_metrics = self.validate(val_dataloader)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 记录历史
            for task in self.model_config.tasks:
                self.train_losses[task].append(train_losses[task])
                self.val_losses[task].append(val_losses[task])
                self.val_metrics[task].append(val_metrics.get(f'{task}_auc', 0.0))
            
            # 早停
            current_val_loss = sum(val_losses.values())
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
            
            # 打印进度
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {sum(train_losses.values()):.4f}, "
                      f"Val Loss = {current_val_loss:.4f}")
            
            # 早停检查
            if patience_counter >= self.model_config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        return self.train_losses, self.val_losses, self.val_metrics