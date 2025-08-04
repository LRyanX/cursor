"""
工具函数模块
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ParetoLoss(nn.Module):
    """Pareto Loss for multi-task learning to avoid seesaw effect"""
    def __init__(self, tasks: List[str], task_weights: Dict[str, float] = None):
        super(ParetoLoss, self).__init__()
        self.tasks = tasks
        self.task_weights = task_weights or {task: 1.0 for task in tasks}
        self.focal_losses = {task: FocalLoss() for task in tasks}
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        losses = {}
        for task in self.tasks:
            if task in predictions and task in targets:
                loss = self.focal_losses[task](predictions[task], targets[task])
                losses[task] = loss * self.task_weights[task]
        
        # Pareto optimal combination
        total_loss = sum(losses.values())
        return total_loss, losses

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_name: str = "") -> Dict[str, float]:
    """计算评估指标"""
    metrics = {}
    
    # AUC
    try:
        metrics[f'{task_name}_auc'] = roc_auc_score(y_true, y_pred)
    except:
        metrics[f'{task_name}_auc'] = 0.5
    
    # Log Loss
    metrics[f'{task_name}_logloss'] = log_loss(y_true, y_pred)
    
    # Brier Score
    metrics[f'{task_name}_brier'] = brier_score_loss(y_true, y_pred)
    
    # Calibration Error
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred, n_bins=10)
        metrics[f'{task_name}_calibration_error'] = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
    except:
        metrics[f'{task_name}_calibration_error'] = 0.0
    
    return metrics

class Calibrator:
    """模型校准器"""
    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.calibrator = None
        
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray):
        """训练校准器"""
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        elif self.method == "platt":
            self.calibrator = LogisticRegression()
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        self.calibrator.fit(y_pred.reshape(-1, 1), y_true)
        return self
    
    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        """校准预测值"""
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted yet")
        
        calibrated_pred = self.calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1]
        return np.clip(calibrated_pred, 0.0, 1.0)

def create_scenario_embeddings(data: pd.DataFrame, scenario_features: List[str], 
                             embedding_dim: int = 8) -> Tuple[np.ndarray, Dict]:
    """创建场景嵌入"""
    scenario_embeddings = {}
    scenario_vectors = []
    
    for feature in scenario_features:
        if feature in data.columns:
            # 简单的哈希嵌入
            unique_values = data[feature].unique()
            embedding_dict = {val: np.random.randn(embedding_dim) for val in unique_values}
            scenario_embeddings[feature] = embedding_dict
            
            # 创建嵌入向量
            feature_embeddings = np.array([embedding_dict.get(val, np.zeros(embedding_dim)) 
                                         for val in data[feature]])
            scenario_vectors.append(feature_embeddings)
    
    if scenario_vectors:
        combined_scenario_vector = np.concatenate(scenario_vectors, axis=1)
    else:
        combined_scenario_vector = np.zeros((len(data), embedding_dim))
    
    return combined_scenario_vector, scenario_embeddings

def balance_sample_weights(y: np.ndarray, task_name: str, 
                         sample_weights: Dict[str, float] = None) -> np.ndarray:
    """平衡样本权重"""
    if sample_weights is None:
        sample_weights = {task_name: 1.0}
    
    weights = np.ones(len(y))
    pos_weight = sample_weights.get(task_name, 1.0)
    
    # 根据正负样本比例调整权重
    pos_ratio = y.mean()
    if pos_ratio > 0:
        neg_weight = 1.0
        pos_weight_adjusted = pos_weight / pos_ratio
        weights[y == 1] = pos_weight_adjusted
        weights[y == 0] = neg_weight
    
    return weights

def feature_importance_analysis(model, feature_names: List[str], 
                              X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """特征重要性分析"""
    importance_scores = {}
    
    # 基于梯度的特征重要性
    if hasattr(model, 'feature_importance'):
        importance_scores = dict(zip(feature_names, model.feature_importance))
    else:
        # 简单的排列重要性
        baseline_score = roc_auc_score(y, model.predict(X))
        for i, feature in enumerate(feature_names):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            permuted_score = roc_auc_score(y, model.predict(X_permuted))
            importance_scores[feature] = baseline_score - permuted_score
    
    return importance_scores

def plot_calibration_curves(y_true_dict: Dict[str, np.ndarray], 
                           y_pred_dict: Dict[str, np.ndarray], 
                           save_path: str = None):
    """绘制校准曲线"""
    fig, axes = plt.subplots(1, len(y_true_dict), figsize=(5*len(y_true_dict), 4))
    if len(y_true_dict) == 1:
        axes = [axes]
    
    for i, (task_name, y_true) in enumerate(y_true_dict.items()):
        y_pred = y_pred_dict[task_name]
        
        # 计算校准曲线
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred, n_bins=10)
        
        # 绘制校准曲线
        axes[i].plot(mean_predicted_value, fraction_of_positives, 'o-', label=f'{task_name}')
        axes[i].plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
        axes[i].set_xlabel('Mean Predicted Probability')
        axes[i].set_ylabel('Fraction of Positives')
        axes[i].set_title(f'{task_name} Calibration Curve')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_curves(train_losses: Dict[str, List[float]], 
                        val_losses: Dict[str, List[float]], 
                        save_path: str = None):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, len(train_losses), figsize=(5*len(train_losses), 4))
    if len(train_losses) == 1:
        axes = [axes]
    
    for i, (task_name, train_loss) in enumerate(train_losses.items()):
        val_loss = val_losses[task_name]
        
        axes[i].plot(train_loss, label=f'{task_name} Train')
        axes[i].plot(val_loss, label=f'{task_name} Val')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        axes[i].set_title(f'{task_name} Training Curve')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def early_stopping(val_losses: List[float], patience: int = 10) -> bool:
    """早停机制"""
    if len(val_losses) < patience:
        return False
    
    best_loss = min(val_losses[:-patience])
    current_loss = val_losses[-1]
    
    return current_loss > best_loss

def save_model_checkpoint(model, optimizer, epoch, loss, path: str):
    """保存模型检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_model_checkpoint(model, optimizer, path: str):
    """加载模型检查点"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']