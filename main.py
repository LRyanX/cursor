#!/usr/bin/env python3
"""
广告DSP多任务多场景预估模型 - 主程序
使用示例和完整训练流程
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import warnings
from typing import Dict, List, Tuple

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from models.feature_processor import FeatureProcessor
from models.multitask_model import MultiTaskMultiScenarioModel
from utils.data_utils import (
    ImbalanceHandler, DataAugmentation, MultiTaskDataset, DataSplitter,
    create_data_loaders, print_data_statistics
)
from utils.calibration import MultiTaskCalibrator, compute_calibration_metrics, plot_calibration_curve
from trainer import MultiTaskTrainer

warnings.filterwarnings('ignore')


def create_sample_data(n_samples: int = 50000) -> pd.DataFrame:
    """创建示例数据用于演示"""
    
    print(f"创建示例数据 ({n_samples} 样本)...")
    
    np.random.seed(42)
    
    # 基础特征
    data = {
        'hour': np.random.randint(0, 24, n_samples),
        'weekday': np.random.randint(1, 8, n_samples),
        'adv_id': np.random.randint(1, 1000, n_samples),
        'affiliate_id': np.random.randint(1, 500, n_samples),
        'campaign_id': np.random.randint(1, 2000, n_samples),
        'ad_group_id': np.random.randint(1, 5000, n_samples),
        'ad_id': np.random.randint(1, 10000, n_samples),
        'creative_id': np.random.randint(1, 8000, n_samples),
        'feature_1': np.random.randint(1, 100, n_samples),
        'pos': np.random.randint(1, 10, n_samples),
        'instl': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'response_type': np.random.randint(1, 5, n_samples),
        'ad_format': np.random.randint(1, 8, n_samples),
        'os': np.random.choice(['iOS', 'Android', 'Windows'], n_samples, p=[0.4, 0.5, 0.1]),
        'device_make': np.random.choice(['Apple', 'Samsung', 'Huawei', 'Xiaomi', 'Other'], 
                                       n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'bundle_id': np.random.randint(1, 50000, n_samples),
        'country': np.random.choice(['US', 'CN', 'JP', 'KR', 'GB', 'DE', 'FR'], 
                                   n_samples, p=[0.3, 0.25, 0.15, 0.1, 0.08, 0.07, 0.05]),
        'package': np.random.randint(1, 30000, n_samples),
        'category': np.random.randint(1, 50, n_samples),
        'connection_type': np.random.choice(['wifi', '4g', '3g', '5g'], 
                                          n_samples, p=[0.5, 0.3, 0.1, 0.1]),
        'device_model': np.random.randint(1, 5000, n_samples),
        'lang': np.random.choice(['en', 'zh', 'ja', 'ko', 'de', 'fr'], 
                                n_samples, p=[0.4, 0.25, 0.15, 0.1, 0.05, 0.05]),
        'publisher_id': np.random.randint(1, 1000, n_samples),
        'first_ssp': np.random.randint(1, 100, n_samples),
        'last_ssp': np.random.randint(1, 100, n_samples),
        'video_placement': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'is_rewarded': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'offer_id': np.random.randint(1, 20000, n_samples),
        'tag_id': np.random.randint(1, 1000, n_samples),
        'osv': np.random.choice(['14.0', '13.0', '12.0', '11.0', '10.0'], 
                               n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'ua': np.random.randint(1, 100000, n_samples),
        'device_id': np.random.randint(1, 1000000, n_samples),
        'ad_width': np.random.choice([320, 728, 300, 250], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'ad_height': np.random.choice([50, 90, 250, 300], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
    }
    
    # 场景特征 (supply开头特征和ip)
    data.update({
        'supply_developer_id': np.random.randint(1, 5000, n_samples),
        'supply_genreId': np.random.randint(1, 100, n_samples),
        'supply_version': np.random.choice(['1.0', '1.1', '1.2', '2.0', '2.1'], 
                                         n_samples, p=[0.2, 0.2, 0.2, 0.2, 0.2]),
        'supply_minimum_os_version': np.random.choice(['10.0', '11.0', '12.0', '13.0'], 
                                                    n_samples, p=[0.25, 0.25, 0.25, 0.25]),
        'supply_industry_id': np.random.randint(1, 50, n_samples),
        'ip': np.random.randint(1, 1000000, n_samples),
    })
    
    # Demand特征
    data.update({
        'demand_developer_id': np.random.randint(1, 8000, n_samples),
        'demand_genreId': np.random.randint(1, 120, n_samples),
        'demand_version': np.random.choice(['1.0', '1.1', '1.2', '2.0', '2.1'], 
                                         n_samples, p=[0.2, 0.2, 0.2, 0.2, 0.2]),
        'demand_minimum_os_version': np.random.choice(['10.0', '11.0', '12.0', '13.0'], 
                                                    n_samples, p=[0.25, 0.25, 0.25, 0.25]),
        'demand_industry_id': np.random.randint(1, 60, n_samples),
        'is_oem': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    })
    
    df = pd.DataFrame(data)
    
    # 生成相关的目标变量（模拟真实业务逻辑）
    # CTR相对容易预测，CVR和IVR更困难
    
    # 基础概率（受多个特征影响）
    base_ctr_prob = 0.1
    base_cvr_prob = 0.05
    base_ivr_prob = 0.02
    
    # 特征影响因子
    ctr_factors = (
        (df['hour'].isin([10, 11, 14, 15, 20, 21]) * 0.5) +  # 高峰时段
        (df['weekday'].isin([6, 7]) * 0.3) +  # 周末
        (df['pos'] <= 3) * 0.4 +  # 前3位置
        (df['os'] == 'iOS') * 0.2 +  # iOS用户
        (df['connection_type'] == 'wifi') * 0.2 +  # WiFi连接
        (df['is_rewarded'] == 1) * 0.3  # 激励广告
    )
    
    cvr_factors = (
        (df['category'].isin(range(1, 11)) * 0.3) +  # 热门类别
        (df['country'].isin(['US', 'JP', 'KR']) * 0.4) +  # 高转化国家
        (df['ad_format'].isin([1, 2, 3]) * 0.2) +  # 优质格式
        (df['supply_industry_id'] <= 20) * 0.2  # 优质行业
    )
    
    ivr_factors = (
        (df['video_placement'] == 1) * 0.4 +  # 视频展示
        (df['ad_width'] * df['ad_height'] > 50000) * 0.3 +  # 大尺寸
        (df['demand_industry_id'] <= 15) * 0.2  # 优质需求方
    )
    
    # 计算最终概率
    ctr_prob = np.clip(base_ctr_prob + ctr_factors * 0.1, 0.01, 0.5)
    cvr_prob = np.clip(base_cvr_prob + cvr_factors * 0.02, 0.005, 0.2)
    ivr_prob = np.clip(base_ivr_prob + ivr_factors * 0.01, 0.002, 0.1)
    
    # 生成目标变量（引入一些随机性）
    df['ctr'] = np.random.binomial(1, ctr_prob)
    df['cvr'] = np.random.binomial(1, cvr_prob)
    df['ivr'] = np.random.binomial(1, ivr_prob)
    
    # 添加一些噪声和缺失值
    missing_mask = np.random.random(n_samples) < 0.05  # 5%的缺失值
    df.loc[missing_mask, 'device_make'] = np.nan
    
    print(f"示例数据创建完成")
    print(f"CTR正样本比例: {df['ctr'].mean():.3f}")
    print(f"CVR正样本比例: {df['cvr'].mean():.3f}")
    print(f"IVR正样本比例: {df['ivr'].mean():.3f}")
    
    return df


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description='广告DSP多任务多场景预估模型')
    parser.add_argument('--data_path', type=str, default=None, 
                       help='数据文件路径，如果不提供则使用示例数据')
    parser.add_argument('--config_path', type=str, default='config.py',
                       help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='输出目录')
    parser.add_argument('--train_mode', type=str, default='full',
                       choices=['full', 'fast'], help='训练模式：full或fast')
    parser.add_argument('--use_sample_data', action='store_true',
                       help='使用示例数据进行演示')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("广告DSP多任务多场景预估模型")
    print("="*80)
    
    # 配置
    config = {
        'sparse_features': SPARSE_FEATURES,
        'scenario_features': SCENARIO_FEATURES,
        'tasks': TASKS,
        'model_config': MODEL_CONFIG,
        'training_config': TRAINING_CONFIG,
        'imbalance_config': IMBALANCE_CONFIG,
        'multitask_config': MULTITASK_CONFIG,
        'calibration_config': CALIBRATION_CONFIG,
        'regularization_config': REGULARIZATION_CONFIG,
        'optimizer_config': OPTIMIZER_CONFIG,
        'data_config': DATA_CONFIG,
        'paths': PATHS
    }
    
    # 快速训练模式配置调整
    if args.train_mode == 'fast':
        config['training_config']['epochs'] = 10
        config['training_config']['batch_size'] = 1024
        config['model_config']['hidden_dims'] = [256, 128]
        config['model_config']['tower_hidden_dims'] = [128]
        print("快速训练模式：减少epochs和模型复杂度")
    
    # 1. 数据加载和准备
    print("\n1. 数据加载和预处理")
    print("-" * 40)
    
    if args.data_path and os.path.exists(args.data_path):
        print(f"从文件加载数据: {args.data_path}")
        data = pd.read_csv(args.data_path)
    elif args.use_sample_data:
        print("使用示例数据")
        sample_size = 10000 if args.train_mode == 'fast' else 50000
        data = create_sample_data(sample_size)
    else:
        print("未指定数据文件，创建示例数据进行演示")
        sample_size = 10000 if args.train_mode == 'fast' else 50000
        data = create_sample_data(sample_size)
    
    # 目标变量映射
    target_columns = {'ctr': 'ctr', 'cvr': 'cvr', 'ivr': 'ivr'}
    
    # 打印数据统计
    print_data_statistics(data, target_columns)
    
    # 2. 特征处理
    print("\n2. 特征处理")
    print("-" * 40)
    
    feature_config = {
        'sparse_features': config['sparse_features'],
        'scenario_features': config['scenario_features'],
        'max_vocab_size': config['data_config']['max_vocab_size'],
        'min_frequency': config['data_config']['min_frequency'],
        'use_hash_encoding': config['data_config']['use_hash_encoding'],
        'hash_bucket_size': config['data_config']['hash_bucket_size']
    }
    
    feature_processor = FeatureProcessor(feature_config)
    
    # 3. 数据分割
    print("\n3. 数据分割")
    print("-" * 40)
    
    data_splitter = DataSplitter(config['training_config'])
    train_data, val_data, test_data = data_splitter.split_data(data, target_columns)
    
    # 4. 特征编码
    print("\n4. 特征编码")
    print("-" * 40)
    
    # 在训练集上拟合特征处理器
    train_features = feature_processor.fit_transform(train_data)
    val_features = feature_processor.transform(val_data) if len(val_data) > 0 else {}
    test_features = feature_processor.transform(test_data) if len(test_data) > 0 else {}
    
    # 获取特征维度信息
    feature_info = feature_processor.get_feature_info()
    print(f"特征维度: {feature_info}")
    
    # 保存特征处理器
    vocab_dir = os.path.join(args.output_dir, 'vocab')
    feature_processor.save(vocab_dir)
    
    # 场景特征维度
    scenario_feature_dims = {}
    for feature_name in config['scenario_features']:
        if feature_name in feature_processor.feature_dims:
            scenario_feature_dims[feature_name] = feature_processor.feature_dims[feature_name]
    
    # 5. 样本不均衡处理
    print("\n5. 样本不均衡处理")
    print("-" * 40)
    
    imbalance_handler = ImbalanceHandler(config['imbalance_config'])
    
    # 计算类别权重
    train_targets = {}
    for task_name, col_name in target_columns.items():
        if col_name in train_data.columns:
            train_targets[task_name] = train_data[col_name].values
    
    class_weights = imbalance_handler.compute_class_weights(train_targets)
    print(f"类别权重: {class_weights}")
    
    # 6. 创建数据集和数据加载器
    print("\n6. 创建数据集")
    print("-" * 40)
    
    # 准备目标张量
    def prepare_targets(data_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        targets = {}
        for task_name, col_name in target_columns.items():
            if col_name in data_df.columns:
                targets[task_name] = torch.tensor(data_df[col_name].values, dtype=torch.float32)
        return targets
    
    # 准备场景特征
    def prepare_scenario_features(features_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        scenario_features = {}
        for feature_name in config['scenario_features']:
            if feature_name in features_dict:
                scenario_features[feature_name] = features_dict[feature_name]
        return scenario_features if scenario_features else None
    
    # 创建数据集
    train_targets_tensor = prepare_targets(train_data)
    train_scenario_features = prepare_scenario_features(train_features)
    
    # 数据增强
    data_augmentation = DataAugmentation(config['regularization_config'])
    
    train_dataset = MultiTaskDataset(
        features=train_features,
        targets=train_targets_tensor,
        scenario_features=train_scenario_features,
        augmentation=data_augmentation
    )
    
    val_dataset = None
    test_dataset = None
    
    if len(val_data) > 0:
        val_targets_tensor = prepare_targets(val_data)
        val_scenario_features = prepare_scenario_features(val_features)
        val_dataset = MultiTaskDataset(
            features=val_features,
            targets=val_targets_tensor,
            scenario_features=val_scenario_features
        )
    
    if len(test_data) > 0:
        test_targets_tensor = prepare_targets(test_data)
        test_scenario_features = prepare_scenario_features(test_features)
        test_dataset = MultiTaskDataset(
            features=test_features,
            targets=test_targets_tensor,
            scenario_features=test_scenario_features
        )
    
    # 创建数据加载器
    batch_size = config['training_config']['batch_size']
    data_loaders = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=2  # 减少worker数量避免内存问题
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    if val_dataset:
        print(f"验证集大小: {len(val_dataset)}")
    if test_dataset:
        print(f"测试集大小: {len(test_dataset)}")
    
    # 7. 模型训练
    print("\n7. 模型训练")
    print("-" * 40)
    
    # 创建训练器
    trainer = MultiTaskTrainer(config)
    
    # 设置日志
    log_dir = os.path.join(args.output_dir, 'logs')
    trainer.setup_logging(log_dir)
    
    # 准备模型
    trainer.prepare_model(feature_processor.feature_dims, scenario_feature_dims)
    trainer.prepare_optimizer()
    
    # 开始训练
    trainer.train(
        train_loader=data_loaders['train'],
        val_loader=data_loaders.get('val'),
        data_augmentation=data_augmentation
    )
    
    # 8. 模型校准
    print("\n8. 模型校准")
    print("-" * 40)
    
    if val_dataset:
        trainer.calibrate_model(data_loaders['val'])
    
    # 9. 模型评估
    print("\n9. 模型评估")
    print("-" * 40)
    
    if test_dataset:
        print("在测试集上评估模型...")
        test_losses, test_metrics = trainer.evaluate(data_loaders['test'], 'test')
        
        print(f"测试集损失: {test_losses}")
        print(f"测试集指标:")
        for task_name, metrics in test_metrics.items():
            print(f"  {task_name}:")
            for metric_name, value in metrics.items():
                print(f"    {metric_name}: {value:.4f}")
        
        # 获取校准前后的预测
        if trainer.calibrator.is_fitted:
            print("\n校准效果评估...")
            raw_predictions, calibrated_predictions = trainer.predict_with_calibration(
                data_loaders['test'], 'platt'
            )
            
            # 计算校准指标
            test_targets_np = {}
            for task_name in config['tasks']:
                if task_name in test_targets_tensor:
                    test_targets_np[task_name] = test_targets_tensor[task_name].numpy()
            
            # 校准前指标
            raw_calibration_metrics = compute_calibration_metrics(raw_predictions, test_targets_np)
            print("校准前指标:")
            for task_name, metrics in raw_calibration_metrics.items():
                print(f"  {task_name}: ECE={metrics['ece']:.4f}, MCE={metrics['mce']:.4f}")
            
            # 校准后指标
            calibrated_calibration_metrics = compute_calibration_metrics(calibrated_predictions, test_targets_np)
            print("校准后指标:")
            for task_name, metrics in calibrated_calibration_metrics.items():
                print(f"  {task_name}: ECE={metrics['ece']:.4f}, MCE={metrics['mce']:.4f}")
            
            # 绘制校准曲线
            try:
                import matplotlib.pyplot as plt
                calibration_plot_path = os.path.join(args.output_dir, 'calibration_curves.png')
                plot_calibration_curve(raw_predictions, test_targets_np, save_path=calibration_plot_path)
                print(f"校准曲线已保存: {calibration_plot_path}")
            except ImportError:
                print("matplotlib未安装，跳过校准曲线绘制")
    
    # 10. 保存最终模型
    print("\n10. 保存模型")
    print("-" * 40)
    
    model_save_path = os.path.join(args.output_dir, 'final_model.pth')
    trainer.save_checkpoint(model_save_path)
    
    print(f"模型已保存到: {model_save_path}")
    print(f"特征处理器已保存到: {vocab_dir}")
    print(f"训练日志已保存到: {log_dir}")
    
    print("\n" + "="*80)
    print("训练完成！")
    print("="*80)
    
    # 11. 模型使用示例
    print("\n11. 模型使用示例")
    print("-" * 40)
    
    # 演示如何加载和使用训练好的模型
    print("演示模型推理...")
    
    # 创建一些示例数据进行推理
    sample_data = data.head(10).copy()
    sample_features = feature_processor.transform(sample_data)
    sample_scenario_features = prepare_scenario_features(sample_features)
    
    sample_dataset = MultiTaskDataset(
        features=sample_features,
        targets={task: torch.zeros(10) for task in config['tasks']},  # 虚拟目标
        scenario_features=sample_scenario_features
    )
    
    sample_loader = torch.utils.data.DataLoader(sample_dataset, batch_size=10, shuffle=False)
    
    # 获取预测
    trainer.model.eval()
    with torch.no_grad():
        for batch in sample_loader:
            features = batch['features']
            scenario_features = batch['scenario_features']
            
            # 移动到设备
            for key in features:
                features[key] = features[key].to(trainer.device)
            if scenario_features is not None:
                for key in scenario_features:
                    scenario_features[key] = scenario_features[key].to(trainer.device)
            
            predictions = trainer.model(features, scenario_features)
            
            print("样本预测结果:")
            for i in range(min(5, len(predictions[config['tasks'][0]]))):  # 只显示前5个样本
                print(f"  样本 {i+1}:")
                for task_name in config['tasks']:
                    pred_value = predictions[task_name][i].item()
                    print(f"    {task_name}: {pred_value:.4f}")
            break
    
    print("\n模型训练和评估完成！")
    

if __name__ == '__main__':
    main()