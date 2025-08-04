import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import logging
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from multi_task_dsp_model import MultiTaskMultiSceneDSPModel
from data_utils import DataProcessor

class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self, tasks: List[str]):
        self.tasks = tasks
        
    def calculate_metrics(self, 
                         predictions: Dict[str, np.ndarray], 
                         targets: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """计算各种评估指标"""
        metrics = {}
        
        for task in self.tasks:
            if task not in predictions or task not in targets:
                continue
                
            pred = predictions[task]
            true = targets[task]
            
            # 处理缺失值
            valid_mask = ~np.isnan(true)
            if not np.any(valid_mask):
                metrics[task] = {'auc': 0.0, 'pr_auc': 0.0, 'logloss': float('inf')}
                continue
                
            pred_valid = pred[valid_mask]
            true_valid = true[valid_mask]
            
            task_metrics = {}
            
            try:
                # AUC
                if len(np.unique(true_valid)) > 1:
                    task_metrics['auc'] = roc_auc_score(true_valid, pred_valid)
                    
                    # PR-AUC
                    precision, recall, _ = precision_recall_curve(true_valid, pred_valid)
                    task_metrics['pr_auc'] = auc(recall, precision)
                else:
                    task_metrics['auc'] = 0.0
                    task_metrics['pr_auc'] = 0.0
                
                # Log Loss
                pred_clipped = np.clip(pred_valid, 1e-7, 1 - 1e-7)
                task_metrics['logloss'] = log_loss(true_valid, pred_clipped)
                
                # 校准指标
                task_metrics.update(self._calculate_calibration_metrics(pred_valid, true_valid))
                
            except Exception as e:
                logging.warning(f"Error calculating metrics for task {task}: {e}")
                task_metrics = {'auc': 0.0, 'pr_auc': 0.0, 'logloss': float('inf')}
            
            metrics[task] = task_metrics
        
        return metrics
    
    def _calculate_calibration_metrics(self, pred: np.ndarray, true: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
        """计算校准指标"""
        try:
            # ECE (Expected Calibration Error)
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            ece = 0.0
            mce = 0.0  # Maximum Calibration Error
            
            for i in range(n_bins):
                mask = (pred >= bin_edges[i]) & (pred < bin_edges[i + 1])
                if i == n_bins - 1:  # 最后一个bin包含右边界
                    mask = (pred >= bin_edges[i]) & (pred <= bin_edges[i + 1])
                
                if np.any(mask):
                    bin_acc = np.mean(true[mask])
                    bin_conf = np.mean(pred[mask])
                    bin_size = np.sum(mask)
                    
                    calibration_error = abs(bin_acc - bin_conf)
                    ece += (bin_size / len(pred)) * calibration_error
                    mce = max(mce, calibration_error)
            
            return {'ece': ece, 'mce': mce}
            
        except Exception as e:
            logging.warning(f"Error calculating calibration metrics: {e}")
            return {'ece': float('inf'), 'mce': float('inf')}

class DSPTrainer:
    """DSP模型训练器"""
    
    def __init__(self, 
                 model: MultiTaskMultiSceneDSPModel,
                 device: str = 'cuda',
                 log_dir: str = './logs',
                 checkpoint_dir: str = './checkpoints'):
        """
        Args:
            model: 模型实例
            device: 训练设备
            log_dir: 日志目录
            checkpoint_dir: 检查点目录
        """
        self.model = model
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        
        # 创建目录
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 初始化记录器
        self.writer = SummaryWriter(log_dir)
        self.metrics_calculator = MetricsCalculator(model.tasks)
        
        # 训练历史
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        
        # 将模型移到设备
        self.model.to(device)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        
    def train(self, 
              train_loader,
              val_loader,
              num_epochs: int = 100,
              learning_rate: float = 1e-3,
              weight_decay: float = 1e-5,
              patience: int = 10,
              scheduler_step_size: int = 30,
              scheduler_gamma: float = 0.5) -> Dict:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            learning_rate: 学习率
            weight_decay: 权重衰减
            patience: 早停耐心值
            scheduler_step_size: 学习率调度步长
            scheduler_gamma: 学习率衰减因子
            
        Returns:
            训练历史字典
        """
        # 设置优化器和调度器
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=scheduler_step_size, 
            gamma=scheduler_gamma
        )
        
        # 早停相关变量
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        logging.info(f"开始训练，总共 {num_epochs} 轮")
        
        for epoch in range(num_epochs):
            # 训练阶段
            train_metrics = self._train_epoch(train_loader, optimizer)
            
            # 验证阶段
            val_metrics = self._validate_epoch(val_loader)
            
            # 更新学习率
            scheduler.step()
            
            # 记录指标
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # 早停检查
            current_val_loss = val_metrics['total_loss']
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint(epoch, 'best_model.pth')
            else:
                patience_counter += 1
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, f'checkpoint_epoch_{epoch+1}.pth')
            
            logging.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['total_loss']:.4f}, "
                f"Val Loss: {val_metrics['total_loss']:.4f}, "
                f"Best Epoch: {best_epoch+1}"
            )
            
            # 早停
            if patience_counter >= patience:
                logging.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        # 加载最佳模型
        self._load_checkpoint('best_model.pth')
        
        logging.info(f"训练完成，最佳验证损失: {best_val_loss:.4f} (第 {best_epoch+1} 轮)")
        
        return {
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history),
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss
        }
    
    def _train_epoch(self, train_loader, optimizer) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        task_losses = defaultdict(float)
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # 移动数据到设备
            dense_features = batch['dense_features'].to(self.device)
            sparse_features = batch['sparse_features'].to(self.device)
            scene_ids = batch['scene_id'].to(self.device)
            
            # 准备目标和权重
            targets = {}
            sample_weights = {}
            
            for task in self.model.tasks:
                if f'target_{task}' in batch:
                    targets[task] = batch[f'target_{task}'].to(self.device)
                if f'weight_{task}' in batch:
                    sample_weights[task] = batch[f'weight_{task}'].to(self.device)
            
            # 前向传播
            predictions = self.model(dense_features, sparse_features, scene_ids)
            
            # 计算损失
            loss, batch_task_losses = self.model.compute_loss(
                predictions, targets, sample_weights
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            for task, task_loss in batch_task_losses.items():
                task_losses[task] += task_loss
            
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}"
            })
        
        # 计算平均损失
        avg_metrics = {'total_loss': total_loss / num_batches}
        for task, loss_sum in task_losses.items():
            avg_metrics[f'{task}_loss'] = loss_sum / num_batches
        
        return avg_metrics
    
    def _validate_epoch(self, val_loader) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0
        task_losses = defaultdict(float)
        all_predictions = defaultdict(list)
        all_targets = defaultdict(list)
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # 移动数据到设备
                dense_features = batch['dense_features'].to(self.device)
                sparse_features = batch['sparse_features'].to(self.device)
                scene_ids = batch['scene_id'].to(self.device)
                
                # 准备目标
                targets = {}
                for task in self.model.tasks:
                    if f'target_{task}' in batch:
                        targets[task] = batch[f'target_{task}'].to(self.device)
                
                # 前向传播
                predictions = self.model(dense_features, sparse_features, scene_ids)
                
                # 计算损失
                loss, batch_task_losses = self.model.compute_loss(predictions, targets)
                
                # 记录损失
                total_loss += loss.item()
                for task, task_loss in batch_task_losses.items():
                    task_losses[task] += task_loss
                
                # 收集预测和目标用于指标计算
                for task in self.model.tasks:
                    if task in predictions and task in targets:
                        all_predictions[task].extend(predictions[task].cpu().numpy())
                        all_targets[task].extend(targets[task].cpu().numpy())
                
                num_batches += 1
        
        # 计算平均损失
        avg_metrics = {'total_loss': total_loss / num_batches}
        for task, loss_sum in task_losses.items():
            avg_metrics[f'{task}_loss'] = loss_sum / num_batches
        
        # 计算评估指标
        pred_arrays = {}
        target_arrays = {}
        for task in all_predictions:
            pred_arrays[task] = np.array(all_predictions[task])
            target_arrays[task] = np.array(all_targets[task])
        
        eval_metrics = self.metrics_calculator.calculate_metrics(pred_arrays, target_arrays)
        
        # 合并指标
        for task, task_metrics in eval_metrics.items():
            for metric_name, metric_value in task_metrics.items():
                avg_metrics[f'{task}_{metric_name}'] = metric_value
        
        return avg_metrics
    
    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """记录指标到tensorboard和历史"""
        # 记录到tensorboard
        for metric_name, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{metric_name}', value, epoch)
            self.train_history[metric_name].append(value)
        
        for metric_name, value in val_metrics.items():
            self.writer.add_scalar(f'Val/{metric_name}', value, epoch)
            self.val_history[metric_name].append(value)
        
        # 记录学习率
        current_lr = self.model.parameters().__next__().grad
        if current_lr is not None:
            for param_group in self.model.parameters():
                if hasattr(param_group, 'lr'):
                    self.writer.add_scalar('Learning_Rate', param_group.lr, epoch)
                    break
    
    def _save_checkpoint(self, epoch: int, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history)
        }
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        logging.info(f"检查点已保存: {filepath}")
    
    def _load_checkpoint(self, filename: str):
        """加载检查点"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"检查点已加载: {filepath}")
        else:
            logging.warning(f"检查点文件不存在: {filepath}")
    
    def evaluate(self, test_loader) -> Dict:
        """评估模型"""
        self.model.eval()
        
        all_predictions = defaultdict(list)
        all_targets = defaultdict(list)
        all_scene_ids = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                dense_features = batch['dense_features'].to(self.device)
                sparse_features = batch['sparse_features'].to(self.device)
                scene_ids = batch['scene_id'].to(self.device)
                
                # 预测
                predictions = self.model(dense_features, sparse_features, scene_ids)
                
                # 收集结果
                all_scene_ids.extend(scene_ids.cpu().numpy())
                
                for task in self.model.tasks:
                    if task in predictions:
                        all_predictions[task].extend(predictions[task].cpu().numpy())
                    
                    if f'target_{task}' in batch:
                        all_targets[task].extend(batch[f'target_{task}'].numpy())
        
        # 转换为numpy数组
        pred_arrays = {}
        target_arrays = {}
        for task in all_predictions:
            pred_arrays[task] = np.array(all_predictions[task])
            if task in all_targets:
                target_arrays[task] = np.array(all_targets[task])
        
        # 计算整体指标
        overall_metrics = self.metrics_calculator.calculate_metrics(pred_arrays, target_arrays)
        
        # 按场景计算指标
        scene_metrics = self._calculate_scene_metrics(
            pred_arrays, target_arrays, np.array(all_scene_ids)
        )
        
        return {
            'overall_metrics': overall_metrics,
            'scene_metrics': scene_metrics,
            'predictions': pred_arrays,
            'targets': target_arrays
        }
    
    def _calculate_scene_metrics(self, 
                                predictions: Dict[str, np.ndarray], 
                                targets: Dict[str, np.ndarray], 
                                scene_ids: np.ndarray) -> Dict:
        """按场景计算指标"""
        scene_metrics = {}
        unique_scenes = np.unique(scene_ids)
        
        for scene_id in unique_scenes:
            scene_mask = scene_ids == scene_id
            
            scene_predictions = {}
            scene_targets = {}
            
            for task in predictions:
                scene_predictions[task] = predictions[task][scene_mask]
                if task in targets:
                    scene_targets[task] = targets[task][scene_mask]
            
            if scene_targets:
                scene_metrics[f'scene_{scene_id}'] = self.metrics_calculator.calculate_metrics(
                    scene_predictions, scene_targets
                )
        
        return scene_metrics
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Curves', fontsize=16)
        
        # 损失曲线
        ax = axes[0, 0]
        if 'total_loss' in self.train_history:
            ax.plot(self.train_history['total_loss'], label='Train', alpha=0.8)
        if 'total_loss' in self.val_history:
            ax.plot(self.val_history['total_loss'], label='Validation', alpha=0.8)
        ax.set_title('Total Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # AUC曲线
        ax = axes[0, 1]
        for task in self.model.tasks:
            metric_name = f'{task}_auc'
            if metric_name in self.val_history:
                ax.plot(self.val_history[metric_name], label=f'{task.upper()} AUC', alpha=0.8)
        ax.set_title('AUC Scores')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 各任务损失
        ax = axes[1, 0]
        for task in self.model.tasks:
            loss_name = f'{task}_loss'
            if loss_name in self.val_history:
                ax.plot(self.val_history[loss_name], label=f'{task.upper()} Loss', alpha=0.8)
        ax.set_title('Task Losses')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 校准误差
        ax = axes[1, 1]
        for task in self.model.tasks:
            ece_name = f'{task}_ece'
            if ece_name in self.val_history:
                ax.plot(self.val_history[ece_name], label=f'{task.upper()} ECE', alpha=0.8)
        ax.set_title('Calibration Error (ECE)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('ECE')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"训练曲线已保存: {save_path}")
        
        plt.show()
    
    def save_training_history(self, filepath: str):
        """保存训练历史"""
        history = {
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history)
        }
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        
        logging.info(f"训练历史已保存: {filepath}")
    
    def close(self):
        """关闭trainer"""
        self.writer.close()
        logging.info("Trainer已关闭")