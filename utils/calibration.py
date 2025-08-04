"""
指标校准模块：实现多种校准方法确保预估值与真实值接近
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.calibration import CalibratedClassifierCV, isotonic_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns


class PlattScaling:
    """Platt Scaling校准"""
    
    def __init__(self):
        self.calibrators = {}
        self.is_fitted = False
    
    def fit(self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]):
        """拟合Platt Scaling参数"""
        
        for task_name in predictions.keys():
            if task_name in targets:
                pred = predictions[task_name]
                target = targets[task_name]
                
                # 将概率转换为logits
                pred_logits = np.log(pred / (1 - pred + 1e-8) + 1e-8)
                
                # 拟合逻辑回归
                calibrator = LogisticRegression()
                calibrator.fit(pred_logits.reshape(-1, 1), target)
                
                self.calibrators[task_name] = calibrator
        
        self.is_fitted = True
    
    def transform(self, predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """应用Platt Scaling校准"""
        
        if not self.is_fitted:
            raise ValueError("PlattScaling must be fitted before transform")
        
        calibrated_predictions = {}
        
        for task_name, pred in predictions.items():
            if task_name in self.calibrators:
                pred_logits = np.log(pred / (1 - pred + 1e-8) + 1e-8)
                calibrated_pred = self.calibrators[task_name].predict_proba(pred_logits.reshape(-1, 1))[:, 1]
                calibrated_predictions[task_name] = calibrated_pred
            else:
                calibrated_predictions[task_name] = pred
        
        return calibrated_predictions


class IsotonicCalibration:
    """等渗回归校准"""
    
    def __init__(self):
        self.calibrators = {}
        self.is_fitted = False
    
    def fit(self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]):
        """拟合等渗回归参数"""
        
        for task_name in predictions.keys():
            if task_name in targets:
                pred = predictions[task_name]
                target = targets[task_name]
                
                # 使用sklearn的等渗回归
                calibrated_pred = isotonic_regression(target, pred)
                
                # 存储映射关系
                self.calibrators[task_name] = {
                    'original_pred': pred,
                    'calibrated_pred': calibrated_pred,
                    'target': target
                }
        
        self.is_fitted = True
    
    def transform(self, predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """应用等渗回归校准"""
        
        if not self.is_fitted:
            raise ValueError("IsotonicCalibration must be fitted before transform")
        
        calibrated_predictions = {}
        
        for task_name, pred in predictions.items():
            if task_name in self.calibrators:
                calibrator = self.calibrators[task_name]
                original_pred = calibrator['original_pred']
                calibrated_pred = calibrator['calibrated_pred']
                
                # 插值获取校准后的预测值
                calibrated = np.interp(pred, original_pred, calibrated_pred)
                calibrated_predictions[task_name] = calibrated
            else:
                calibrated_predictions[task_name] = pred
        
        return calibrated_predictions


class TemperatureScaling(nn.Module):
    """温度缩放校准"""
    
    def __init__(self, num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks
        self.temperatures = nn.Parameter(torch.ones(num_tasks))
        self.is_fitted = False
    
    def forward(self, logits: List[torch.Tensor]) -> List[torch.Tensor]:
        """应用温度缩放"""
        
        calibrated_logits = []
        for i, logit in enumerate(logits):
            calibrated_logit = logit / self.temperatures[i]
            calibrated_logits.append(calibrated_logit)
        
        return calibrated_logits
    
    def fit(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
            task_names: List[str], max_iter: int = 1000, lr: float = 0.01):
        """拟合温度参数"""
        
        # 将预测转换为logits
        logits_list = []
        targets_list = []
        
        for i, task_name in enumerate(task_names):
            if task_name in predictions and task_name in targets:
                pred = predictions[task_name]
                target = targets[task_name]
                
                # 概率转logits
                logit = torch.log(pred / (1 - pred + 1e-8) + 1e-8)
                logits_list.append(logit)
                targets_list.append(target)
        
        if not logits_list:
            return
        
        optimizer = torch.optim.Adam([self.temperatures], lr=lr)
        best_loss = float('inf')
        best_temperatures = self.temperatures.clone()
        
        for epoch in range(max_iter):
            optimizer.zero_grad()
            
            total_loss = 0
            calibrated_logits = self.forward(logits_list)
            
            for i, (calibrated_logit, target) in enumerate(zip(calibrated_logits, targets_list)):
                calibrated_prob = torch.sigmoid(calibrated_logit)
                loss = F.binary_cross_entropy(calibrated_prob, target)
                total_loss += loss
            
            total_loss.backward()
            optimizer.step()
            
            # 保存最佳参数
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_temperatures = self.temperatures.clone()
            
            # 约束温度为正值
            with torch.no_grad():
                self.temperatures.clamp_(min=0.1, max=10.0)
        
        # 恢复最佳参数
        self.temperatures.data = best_temperatures
        self.is_fitted = True
    
    def get_temperatures(self) -> List[float]:
        """获取温度参数"""
        return self.temperatures.detach().cpu().numpy().tolist()


class BinningCalibration:
    """分箱校准"""
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.calibration_maps = {}
        self.is_fitted = False
    
    def fit(self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]):
        """拟合分箱校准参数"""
        
        for task_name in predictions.keys():
            if task_name in targets:
                pred = predictions[task_name]
                target = targets[task_name]
                
                # 创建分箱
                bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]
                
                calibration_map = {}
                
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    # 找到在当前箱中的样本
                    in_bin = (pred > bin_lower) & (pred <= bin_upper)
                    
                    if in_bin.sum() > 0:
                        # 计算该箱的真实正样本比例
                        true_positive_rate = target[in_bin].mean()
                        calibration_map[(bin_lower, bin_upper)] = true_positive_rate
                    else:
                        # 如果箱中没有样本，使用箱中点作为预测值
                        calibration_map[(bin_lower, bin_upper)] = (bin_lower + bin_upper) / 2
                
                self.calibration_maps[task_name] = calibration_map
        
        self.is_fitted = True
    
    def transform(self, predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """应用分箱校准"""
        
        if not self.is_fitted:
            raise ValueError("BinningCalibration must be fitted before transform")
        
        calibrated_predictions = {}
        
        for task_name, pred in predictions.items():
            if task_name in self.calibration_maps:
                calibration_map = self.calibration_maps[task_name]
                calibrated_pred = np.zeros_like(pred)
                
                for (bin_lower, bin_upper), true_rate in calibration_map.items():
                    in_bin = (pred > bin_lower) & (pred <= bin_upper)
                    calibrated_pred[in_bin] = true_rate
                
                calibrated_predictions[task_name] = calibrated_pred
            else:
                calibrated_predictions[task_name] = pred
        
        return calibrated_predictions


class MultiTaskCalibrator:
    """多任务校准器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.use_platt_scaling = config.get('use_platt_scaling', True)
        self.use_isotonic_regression = config.get('use_isotonic_regression', True)
        self.use_temperature_scaling = config.get('temperature_scaling', True)
        self.calibration_bins = config.get('calibration_bins', 10)
        
        # 初始化校准器
        self.platt_scaler = PlattScaling() if self.use_platt_scaling else None
        self.isotonic_calibrator = IsotonicCalibration() if self.use_isotonic_regression else None
        self.temperature_scaler = None
        self.binning_calibrator = BinningCalibration(self.calibration_bins)
        
        self.is_fitted = False
    
    def fit(self, predictions: Dict[str, Union[np.ndarray, torch.Tensor]], 
            targets: Dict[str, Union[np.ndarray, torch.Tensor]],
            task_names: List[str]):
        """拟合所有校准器"""
        
        # 转换为numpy数组
        pred_np = {}
        target_np = {}
        
        for task_name in task_names:
            if task_name in predictions and task_name in targets:
                pred = predictions[task_name]
                target = targets[task_name]
                
                if isinstance(pred, torch.Tensor):
                    pred = pred.detach().cpu().numpy()
                if isinstance(target, torch.Tensor):
                    target = target.detach().cpu().numpy()
                
                pred_np[task_name] = pred
                target_np[task_name] = target
        
        # 拟合各种校准器
        if self.platt_scaler is not None:
            self.platt_scaler.fit(pred_np, target_np)
        
        if self.isotonic_calibrator is not None:
            self.isotonic_calibrator.fit(pred_np, target_np)
        
        if self.use_temperature_scaling:
            # 转换回torch tensor进行温度缩放
            pred_torch = {k: torch.tensor(v, dtype=torch.float32) for k, v in pred_np.items()}
            target_torch = {k: torch.tensor(v, dtype=torch.float32) for k, v in target_np.items()}
            
            self.temperature_scaler = TemperatureScaling(len(task_names))
            self.temperature_scaler.fit(pred_torch, target_torch, task_names)
        
        self.binning_calibrator.fit(pred_np, target_np)
        
        self.is_fitted = True
    
    def calibrate(self, predictions: Dict[str, Union[np.ndarray, torch.Tensor]], 
                  method: str = 'platt') -> Dict[str, np.ndarray]:
        """应用校准"""
        
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before use")
        
        # 转换为numpy
        pred_np = {}
        for task_name, pred in predictions.items():
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy()
            pred_np[task_name] = pred
        
        if method == 'platt' and self.platt_scaler is not None:
            return self.platt_scaler.transform(pred_np)
        elif method == 'isotonic' and self.isotonic_calibrator is not None:
            return self.isotonic_calibrator.transform(pred_np)
        elif method == 'binning':
            return self.binning_calibrator.transform(pred_np)
        elif method == 'temperature' and self.temperature_scaler is not None:
            # 温度缩放需要特殊处理
            pred_torch = {k: torch.tensor(v, dtype=torch.float32) for k, v in pred_np.items()}
            logits = [torch.log(pred / (1 - pred + 1e-8) + 1e-8) for pred in pred_torch.values()]
            calibrated_logits = self.temperature_scaler(logits)
            calibrated_probs = [torch.sigmoid(logit) for logit in calibrated_logits]
            
            result = {}
            for i, task_name in enumerate(pred_torch.keys()):
                result[task_name] = calibrated_probs[i].detach().cpu().numpy()
            return result
        else:
            return pred_np


def compute_calibration_metrics(predictions: Dict[str, np.ndarray], 
                               targets: Dict[str, np.ndarray],
                               n_bins: int = 10) -> Dict[str, Dict[str, float]]:
    """计算校准指标"""
    
    results = {}
    
    for task_name in predictions.keys():
        if task_name in targets:
            pred = predictions[task_name]
            target = targets[task_name]
            
            # Expected Calibration Error (ECE)
            ece = expected_calibration_error(pred, target, n_bins)
            
            # Maximum Calibration Error (MCE)  
            mce = maximum_calibration_error(pred, target, n_bins)
            
            # Brier Score
            brier = brier_score_loss(target, pred)
            
            # Log Loss
            logloss = log_loss(target, pred)
            
            results[task_name] = {
                'ece': ece,
                'mce': mce,
                'brier_score': brier,
                'log_loss': logloss
            }
    
    return results


def expected_calibration_error(predictions: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """计算Expected Calibration Error"""
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    total_samples = len(predictions)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 找到在当前箱中的样本
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # 计算当前箱的准确率和置信度
            accuracy_in_bin = targets[in_bin].mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            
            # 计算该箱对ECE的贡献
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def maximum_calibration_error(predictions: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """计算Maximum Calibration Error"""
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    mce = 0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 找到在当前箱中的样本
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        
        if in_bin.sum() > 0:
            # 计算当前箱的准确率和置信度
            accuracy_in_bin = targets[in_bin].mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            
            # 更新最大校准误差
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
    
    return mce


def plot_calibration_curve(predictions: Dict[str, np.ndarray], 
                          targets: Dict[str, np.ndarray],
                          n_bins: int = 10,
                          save_path: Optional[str] = None):
    """绘制校准曲线"""
    
    fig, axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 5))
    if len(predictions) == 1:
        axes = [axes]
    
    for i, (task_name, pred) in enumerate(predictions.items()):
        if task_name in targets:
            target = targets[task_name]
            
            # 计算分箱统计
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            bin_centers = []
            accuracies = []
            counts = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (pred > bin_lower) & (pred <= bin_upper)
                if in_bin.sum() > 0:
                    bin_centers.append((bin_lower + bin_upper) / 2)
                    accuracies.append(target[in_bin].mean())
                    counts.append(in_bin.sum())
            
            # 绘制校准曲线
            ax = axes[i]
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            ax.scatter(bin_centers, accuracies, s=[c/10 for c in counts], alpha=0.7, label='Observed')
            ax.plot(bin_centers, accuracies, 'r-', alpha=0.7)
            
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title(f'Calibration Curve - {task_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def reliability_diagram(predictions: Dict[str, np.ndarray],
                       targets: Dict[str, np.ndarray],
                       n_bins: int = 10,
                       save_path: Optional[str] = None):
    """绘制可靠性图"""
    
    fig, axes = plt.subplots(2, len(predictions), figsize=(5 * len(predictions), 10))
    if len(predictions) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (task_name, pred) in enumerate(predictions.items()):
        if task_name in targets:
            target = targets[task_name]
            
            # 计算分箱统计
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            bin_centers = []
            accuracies = []
            confidences = []
            counts = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (pred > bin_lower) & (pred <= bin_upper)
                if in_bin.sum() > 0:
                    bin_centers.append((bin_lower + bin_upper) / 2)
                    accuracies.append(target[in_bin].mean())
                    confidences.append(pred[in_bin].mean())
                    counts.append(in_bin.sum())
            
            # 上图：可靠性图
            ax1 = axes[0, i]
            bars = ax1.bar(bin_centers, accuracies, width=0.08, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            ax1.plot(bin_centers, confidences, 'ro-', label='Mean Prediction')
            ax1.set_xlabel('Confidence')
            ax1.set_ylabel('Accuracy')
            ax1.set_title(f'Reliability Diagram - {task_name}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 下图：样本分布
            ax2 = axes[1, i]
            ax2.bar(bin_centers, counts, width=0.08, alpha=0.7, color='lightcoral', edgecolor='black')
            ax2.set_xlabel('Confidence')
            ax2.set_ylabel('Count')
            ax2.set_title(f'Sample Distribution - {task_name}')
            ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig