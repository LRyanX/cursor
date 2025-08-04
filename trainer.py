"""
训练器模块：完整的训练流程，包含早停、学习率调度、多任务权重动态调整
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import os
import time
from tqdm import tqdm
import json
import warnings
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, log_loss

from models.multitask_model import MultiTaskMultiScenarioModel
from models.feature_processor import FeatureProcessor
from utils.data_utils import ImbalanceHandler, DataAugmentation, MultiTaskDataset
from utils.calibration import MultiTaskCalibrator, compute_calibration_metrics
from config import *


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 monitor: str = 'val_loss', mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        self.best_epoch = 0
        self.wait = 0
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
            self.min_delta *= 1
    
    def __call__(self, current_score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
        elif self.monitor_op(current_score, self.best_score + self.min_delta):
            self.best_score = current_score
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class GradientNormalization:
    """梯度归一化，解决多任务训练中的跷跷板问题"""
    
    def __init__(self, alpha: float = 0.16):
        self.alpha = alpha
        self.task_losses_history = []
    
    def normalize_gradients(self, model: nn.Module, task_losses: Dict[str, torch.Tensor], 
                          tasks: List[str]) -> Dict[str, float]:
        """归一化各任务的梯度"""
        
        # 计算各任务的梯度范数
        task_grads = {}
        
        for i, task_name in enumerate(tasks):
            # 获取当前任务的损失
            task_loss = task_losses[task_name]
            
            # 清零梯度
            model.zero_grad()
            
            # 反向传播单个任务
            task_loss.backward(retain_graph=True)
            
            # 计算梯度范数
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            task_grads[task_name] = total_norm
        
        # 计算平均梯度范数
        avg_grad_norm = np.mean(list(task_grads.values()))
        
        # 计算归一化权重
        normalized_weights = {}
        for task_name, grad_norm in task_grads.items():
            if grad_norm > 0:
                normalized_weights[task_name] = avg_grad_norm / grad_norm
            else:
                normalized_weights[task_name] = 1.0
        
        return normalized_weights
    
    def update_task_weights(self, current_losses: Dict[str, float], 
                          current_weights: List[float]) -> List[float]:
        """基于损失变化率动态更新任务权重"""
        
        self.task_losses_history.append(current_losses)
        
        if len(self.task_losses_history) < 2:
            return current_weights
        
        # 计算损失变化率
        prev_losses = self.task_losses_history[-2]
        loss_rates = {}
        
        for task_name in current_losses.keys():
            if task_name in prev_losses:
                if prev_losses[task_name] > 0:
                    rate = (current_losses[task_name] - prev_losses[task_name]) / prev_losses[task_name]
                    loss_rates[task_name] = rate
                else:
                    loss_rates[task_name] = 0.0
        
        # 基于损失变化率调整权重
        new_weights = []
        for i, weight in enumerate(current_weights):
            task_name = list(current_losses.keys())[i]
            if task_name in loss_rates:
                # 如果损失下降缓慢，增加权重
                rate = loss_rates[task_name]
                if rate > 0:  # 损失增加
                    new_weight = weight * (1 + self.alpha)
                else:  # 损失减少
                    new_weight = weight * (1 - self.alpha * abs(rate))
                new_weights.append(max(0.1, min(2.0, new_weight)))  # 限制权重范围
            else:
                new_weights.append(weight)
        
        return new_weights


class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self, tasks: List[str]):
        self.tasks = tasks
    
    def compute_metrics(self, predictions: Dict[str, np.ndarray], 
                       targets: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """计算各种评估指标"""
        
        results = {}
        
        for task_name in self.tasks:
            if task_name in predictions and task_name in targets:
                pred = predictions[task_name]
                target = targets[task_name]
                
                # 确保预测值在有效范围内
                pred = np.clip(pred, 1e-7, 1 - 1e-7)
                
                # 计算二分类指标
                pred_binary = (pred > 0.5).astype(int)
                
                try:
                    auc = roc_auc_score(target, pred)
                except ValueError:
                    auc = 0.5  # 如果只有一个类别，AUC为0.5
                
                accuracy = accuracy_score(target, pred_binary)
                precision = precision_score(target, pred_binary, zero_division=0)
                recall = recall_score(target, pred_binary, zero_division=0)
                f1 = f1_score(target, pred_binary, zero_division=0)
                logloss = log_loss(target, pred)
                
                results[task_name] = {
                    'auc': auc,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'logloss': logloss
                }
        
        return results


class MultiTaskTrainer:
    """多任务多场景模型训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.tasks = config['tasks']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 训练配置
        self.training_config = config['training_config']
        self.imbalance_config = config['imbalance_config']
        self.multitask_config = config['multitask_config']
        self.calibration_config = config['calibration_config']
        self.regularization_config = config['regularization_config']
        self.optimizer_config = config['optimizer_config']
        
        # 初始化组件
        self.model = None
        self.feature_processor = None
        self.optimizer = None
        self.scheduler = None
        self.early_stopping = None
        self.gradient_normalizer = None
        self.metrics_calculator = MetricsCalculator(self.tasks)
        self.calibrator = MultiTaskCalibrator(self.calibration_config)
        
        # 训练状态
        self.current_epoch = 0
        self.best_metrics = {}
        self.training_history = []
        
        # 日志
        self.logger = None
        self.log_dir = None
        
    def setup_logging(self, log_dir: str):
        """设置日志"""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # TensorBoard
        self.logger = SummaryWriter(log_dir)
        
        print(f"日志将保存到: {log_dir}")
    
    def prepare_model(self, feature_dims: Dict[str, int], 
                     scenario_feature_dims: Dict[str, int]):
        """准备模型"""
        
        # 创建模型
        model_config = {
            'tasks': self.tasks,
            'model_config': self.config['model_config'],
            'multitask_config': self.multitask_config,
            'calibration_config': self.calibration_config,
            'regularization_config': self.regularization_config
        }
        
        self.model = MultiTaskMultiScenarioModel(model_config)
        self.model.build_model(feature_dims, scenario_feature_dims)
        self.model.to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"模型参数总数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
    def prepare_optimizer(self):
        """准备优化器和学习率调度器"""
        
        # 优化器
        if self.optimizer_config['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.training_config['learning_rate'],
                weight_decay=self.training_config['weight_decay']
            )
        elif self.optimizer_config['optimizer'] == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.training_config['learning_rate'],
                weight_decay=self.training_config['weight_decay']
            )
        elif self.optimizer_config['optimizer'] == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.training_config['learning_rate'],
                momentum=0.9,
                weight_decay=self.training_config['weight_decay']
            )
        
        # 学习率调度器
        if self.optimizer_config['scheduler'] == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.optimizer_config['scheduler_factor'],
                patience=self.optimizer_config['scheduler_patience'],
                min_lr=self.optimizer_config['min_lr']
            )
        elif self.optimizer_config['scheduler'] == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.8
            )
        elif self.optimizer_config['scheduler'] == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config['epochs']
            )
        
        # 早停
        self.early_stopping = EarlyStopping(
            patience=self.training_config['early_stopping_patience'],
            monitor='val_loss',
            mode='min'
        )
        
        # 梯度归一化
        if self.multitask_config['use_gradient_normalization']:
            self.gradient_normalizer = GradientNormalization()
    
    def train_epoch(self, train_loader: DataLoader, 
                   data_augmentation: Optional[DataAugmentation] = None) -> Dict[str, float]:
        """训练一个epoch"""
        
        self.model.train()
        epoch_losses = {task: [] for task in self.tasks}
        epoch_losses['total_loss'] = []
        epoch_losses['reg_loss'] = []
        
        total_batches = len(train_loader)
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        # 动态dropout调度
        if self.regularization_config['dropout_schedule']:
            # 随着训练进行逐渐减少dropout
            progress = self.current_epoch / self.training_config['epochs']
            current_dropout = self.config['model_config']['dropout_rate'] * (1 - progress * 0.5)
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = current_dropout
        
        for batch_idx, batch in enumerate(pbar):
            features = batch['features']
            targets = batch['targets']
            scenario_features = batch['scenario_features']
            
            # 移动到设备
            for key in features:
                features[key] = features[key].to(self.device)
            for key in targets:
                targets[key] = targets[key].to(self.device)
            if scenario_features is not None:
                for key in scenario_features:
                    scenario_features[key] = scenario_features[key].to(self.device)
            
            # 数据增强
            if data_augmentation is not None and np.random.random() < 0.3:
                # 随机应用Mixup
                if np.random.random() < 0.5:
                    # 将特征转换为连续向量进行Mixup
                    embedded_features = self.model.get_embeddings(features)
                    mixed_features, mixed_targets = data_augmentation.mixup(embedded_features, targets)
                    
                    # 使用混合后的嵌入特征进行前向传播
                    # 这里需要特殊处理，因为模型期望字典格式的特征
                    predictions = self._forward_with_embeddings(mixed_features, scenario_features)
                    targets = mixed_targets
                else:
                    predictions = self.model(features, scenario_features)
            else:
                predictions = self.model(features, scenario_features)
            
            # 计算损失
            total_loss, task_losses = self.model.compute_loss(
                predictions, targets, self.imbalance_config
            )
            
            # 梯度归一化
            if self.gradient_normalizer is not None:
                # 计算归一化权重
                normalized_weights = self.gradient_normalizer.normalize_gradients(
                    self.model, task_losses, self.tasks
                )
                
                # 重新计算加权损失
                total_loss = 0
                for i, task_name in enumerate(self.tasks):
                    if task_name in task_losses:
                        weight = normalized_weights.get(task_name, 1.0)
                        total_loss += weight * task_losses[task_name]
                
                # 添加正则化损失
                if 'reg_loss' in task_losses:
                    total_loss += task_losses['reg_loss']
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            if self.training_config['gradient_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config['gradient_clip_norm']
                )
            
            self.optimizer.step()
            
            # 记录损失
            for task_name in self.tasks:
                if task_name in task_losses:
                    epoch_losses[task_name].append(task_losses[task_name].item())
            epoch_losses['total_loss'].append(total_loss.item())
            if 'reg_loss' in task_losses:
                epoch_losses['reg_loss'].append(task_losses['reg_loss'].item())
            
            # 更新进度条
            current_loss = total_loss.item()
            pbar.set_postfix({'loss': f'{current_loss:.4f}'})
        
        # 计算平均损失
        avg_losses = {}
        for key, losses in epoch_losses.items():
            if losses:
                avg_losses[key] = np.mean(losses)
        
        return avg_losses
    
    def _forward_with_embeddings(self, embedded_features: torch.Tensor, 
                                scenario_features: Optional[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """使用已嵌入的特征进行前向传播"""
        
        # 通过共享网络
        shared_output = self.model.shared_network(embedded_features)
        
        # 注意力机制
        attention_input = shared_output.unsqueeze(1)
        attended_output = self.model.attention(attention_input, attention_input, attention_input)
        attended_output = attended_output.squeeze(1)
        
        # 场景感知处理
        if self.model.scenario_embedding is not None and scenario_features is not None:
            scenario_emb = self.model.scenario_embedding(scenario_features)
            task_inputs = []
            for i in range(self.model.num_tasks):
                gated_features = self.model.scenario_gates[i](attended_output, scenario_emb)
                task_inputs.append(gated_features)
        else:
            task_inputs = [attended_output] * self.model.num_tasks
        
        # Cross-stitch
        task_inputs = self.model.cross_stitch(task_inputs)
        
        # 任务特定预测
        task_outputs = {}
        raw_outputs = []
        
        for i, task_name in enumerate(self.tasks):
            task_output = self.model.task_towers[i](task_inputs[i])
            raw_outputs.append(task_output)
            
            if self.model.temperature_parameters is not None:
                task_output = task_output / self.model.temperature_parameters[i]
            
            task_outputs[task_name] = torch.sigmoid(task_output.squeeze(-1))
        
        task_outputs['raw_outputs'] = raw_outputs
        return task_outputs
    
    def evaluate(self, data_loader: DataLoader, phase: str = 'val') -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """评估模型"""
        
        self.model.eval()
        all_predictions = {task: [] for task in self.tasks}
        all_targets = {task: [] for task in self.tasks}
        eval_losses = {task: [] for task in self.tasks}
        eval_losses['total_loss'] = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f'Evaluating {phase}'):
                features = batch['features']
                targets = batch['targets']
                scenario_features = batch['scenario_features']
                
                # 移动到设备
                for key in features:
                    features[key] = features[key].to(self.device)
                for key in targets:
                    targets[key] = targets[key].to(self.device)
                if scenario_features is not None:
                    for key in scenario_features:
                        scenario_features[key] = scenario_features[key].to(self.device)
                
                # 前向传播
                predictions = self.model(features, scenario_features)
                
                # 计算损失
                total_loss, task_losses = self.model.compute_loss(
                    predictions, targets, self.imbalance_config
                )
                
                # 记录损失
                for task_name in self.tasks:
                    if task_name in task_losses:
                        eval_losses[task_name].append(task_losses[task_name].item())
                eval_losses['total_loss'].append(total_loss.item())
                
                # 收集预测和目标
                for task_name in self.tasks:
                    if task_name in predictions and task_name in targets:
                        all_predictions[task_name].extend(
                            predictions[task_name].cpu().numpy()
                        )
                        all_targets[task_name].extend(
                            targets[task_name].cpu().numpy()
                        )
        
        # 计算平均损失
        avg_losses = {}
        for key, losses in eval_losses.items():
            if losses:
                avg_losses[key] = np.mean(losses)
        
        # 转换为numpy数组
        for task_name in self.tasks:
            if all_predictions[task_name]:
                all_predictions[task_name] = np.array(all_predictions[task_name])
                all_targets[task_name] = np.array(all_targets[task_name])
        
        # 计算指标
        metrics = self.metrics_calculator.compute_metrics(all_predictions, all_targets)
        
        return avg_losses, metrics
    
    def save_checkpoint(self, save_path: str, is_best: bool = False):
        """保存检查点"""
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metrics': self.best_metrics,
            'training_history': self.training_history,
            'config': self.config
        }
        
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = save_path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
        
        print(f"检查点已保存: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_metrics = checkpoint['best_metrics']
        self.training_history = checkpoint['training_history']
        
        print(f"检查点已加载: {checkpoint_path}")
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              data_augmentation: Optional[DataAugmentation] = None):
        """完整训练流程"""
        
        print("开始训练...")
        print(f"训练设备: {self.device}")
        print(f"总epochs: {self.training_config['epochs']}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.training_config['epochs']):
            self.current_epoch = epoch
            
            # 训练
            train_losses = self.train_epoch(train_loader, data_augmentation)
            
            # 验证
            val_losses = {}
            val_metrics = {}
            if val_loader is not None:
                val_losses, val_metrics = self.evaluate(val_loader, 'val')
            
            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    monitor_metric = val_losses.get('total_loss', train_losses['total_loss'])
                    self.scheduler.step(monitor_metric)
                else:
                    self.scheduler.step()
            
            # 记录训练历史
            epoch_history = {
                'epoch': epoch,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_metrics': val_metrics,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_history)
            
            # 记录到TensorBoard
            if self.logger:
                for task_name, loss in train_losses.items():
                    self.logger.add_scalar(f'Loss/Train_{task_name}', loss, epoch)
                
                for task_name, loss in val_losses.items():
                    self.logger.add_scalar(f'Loss/Val_{task_name}', loss, epoch)
                
                for task_name, task_metrics in val_metrics.items():
                    for metric_name, metric_value in task_metrics.items():
                        self.logger.add_scalar(f'Metrics/{task_name}_{metric_name}', metric_value, epoch)
                
                self.logger.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 打印进度
            print(f"\nEpoch {epoch + 1}/{self.training_config['epochs']}")
            print(f"训练损失: {train_losses['total_loss']:.4f}")
            if val_losses:
                print(f"验证损失: {val_losses['total_loss']:.4f}")
            
            # 打印各任务指标
            for task_name in self.tasks:
                if task_name in val_metrics:
                    metrics = val_metrics[task_name]
                    print(f"{task_name}: AUC={metrics['auc']:.4f}, ACC={metrics['accuracy']:.4f}")
            
            # 早停检查
            if val_losses:
                monitor_score = val_losses['total_loss']
                if self.early_stopping(monitor_score, epoch):
                    print(f"早停触发！最佳epoch: {self.early_stopping.best_epoch}")
                    break
            
            # 保存检查点
            if self.log_dir:
                checkpoint_path = os.path.join(self.log_dir, f'checkpoint_epoch_{epoch}.pth')
                is_best = (not self.best_metrics or 
                          val_losses.get('total_loss', float('inf')) < self.best_metrics.get('val_loss', float('inf')))
                
                if is_best:
                    self.best_metrics = {
                        'epoch': epoch,
                        'val_loss': val_losses.get('total_loss', float('inf')),
                        'val_metrics': val_metrics
                    }
                
                self.save_checkpoint(checkpoint_path, is_best)
            
            # 动态调整任务权重
            if (self.gradient_normalizer is not None and 
                epoch > 0 and 
                epoch % self.multitask_config['weight_update_frequency'] == 0):
                
                current_losses = {task: train_losses[task] for task in self.tasks if task in train_losses}
                current_weights = self.multitask_config['task_weights']
                new_weights = self.gradient_normalizer.update_task_weights(current_losses, current_weights)
                self.multitask_config['task_weights'] = new_weights
                print(f"任务权重已更新: {new_weights}")
        
        training_time = time.time() - start_time
        print(f"\n训练完成！总耗时: {training_time:.2f}秒")
        
        # 保存训练历史
        if self.log_dir:
            history_path = os.path.join(self.log_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
        
        # 关闭日志
        if self.logger:
            self.logger.close()
    
    def calibrate_model(self, calibration_loader: DataLoader):
        """校准模型"""
        
        print("开始模型校准...")
        
        # 获取校准数据的预测
        self.model.eval()
        all_predictions = {task: [] for task in self.tasks}
        all_targets = {task: [] for task in self.tasks}
        
        with torch.no_grad():
            for batch in tqdm(calibration_loader, desc='收集校准数据'):
                features = batch['features']
                targets = batch['targets']
                scenario_features = batch['scenario_features']
                
                # 移动到设备
                for key in features:
                    features[key] = features[key].to(self.device)
                for key in targets:
                    targets[key] = targets[key].to(self.device)
                if scenario_features is not None:
                    for key in scenario_features:
                        scenario_features[key] = scenario_features[key].to(self.device)
                
                predictions = self.model(features, scenario_features)
                
                for task_name in self.tasks:
                    if task_name in predictions and task_name in targets:
                        all_predictions[task_name].extend(predictions[task_name].cpu().numpy())
                        all_targets[task_name].extend(targets[task_name].cpu().numpy())
        
        # 转换为numpy数组
        for task_name in self.tasks:
            if all_predictions[task_name]:
                all_predictions[task_name] = np.array(all_predictions[task_name])
                all_targets[task_name] = np.array(all_targets[task_name])
        
        # 拟合校准器
        self.calibrator.fit(all_predictions, all_targets, self.tasks)
        
        print("模型校准完成")
    
    def predict_with_calibration(self, data_loader: DataLoader, 
                               calibration_method: str = 'platt') -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """使用校准进行预测"""
        
        # 获取原始预测
        self.model.eval()
        all_predictions = {task: [] for task in self.tasks}
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='预测中'):
                features = batch['features']
                scenario_features = batch['scenario_features']
                
                for key in features:
                    features[key] = features[key].to(self.device)
                if scenario_features is not None:
                    for key in scenario_features:
                        scenario_features[key] = scenario_features[key].to(self.device)
                
                predictions = self.model(features, scenario_features)
                
                for task_name in self.tasks:
                    if task_name in predictions:
                        all_predictions[task_name].extend(predictions[task_name].cpu().numpy())
        
        # 转换为numpy数组
        for task_name in self.tasks:
            if all_predictions[task_name]:
                all_predictions[task_name] = np.array(all_predictions[task_name])
        
        # 应用校准
        calibrated_predictions = self.calibrator.calibrate(all_predictions, calibration_method)
        
        return all_predictions, calibrated_predictions