#!/usr/bin/env python3
"""
DSP广告模型使用示例

展示如何使用真实的dense和sparse特征训练多任务多场景DSP模型
"""

import pandas as pd
import numpy as np
import torch
import logging

from feature_engineering import create_dsp_feature_processor, create_sample_dsp_data, analyze_feature_importance
from multi_task_dsp_model import MultiTaskMultiSceneDSPModel
from data_utils import DataProcessor
from trainer import DSPTrainer
from config import create_sample_config

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("=== DSP广告模型示例 ===")
    
    # 1. 生成示例DSP数据
    print("\n1. 生成示例DSP数据...")
    df = create_sample_dsp_data(num_samples=5000, num_scenes=3)
    print(f"生成数据: {len(df)} 个样本，{len(df.columns)} 个特征")
    
    # 显示数据概览
    print("\n数据概览:")
    print(df.head())
    
    print("\n特征分布:")
    print("Dense特征示例:")
    dense_cols = ['ssp_cnt', 'bid_floor', 'supply_normal_rate', 'demand_normal_rate']
    print(df[dense_cols].describe())
    
    print("\nSparse特征示例:")
    sparse_cols = ['hour', 'weekday', 'os', 'country', 'ad_format']
    for col in sparse_cols:
        if col in df.columns:
            print(f"{col}: {df[col].value_counts().head(3).to_dict()}")
    
    print("\n目标变量分布:")
    for target in ['ctr', 'cvr', 'ivr']:
        if target in df.columns:
            valid_data = df[target].dropna()
            print(f"{target}: 正样本比例 = {valid_data.mean():.4f}, 有效样本 = {len(valid_data)}")
    
    # 2. 特征重要性分析
    print("\n2. 特征重要性分析...")
    
    # 获取特征定义
    processor = create_dsp_feature_processor()
    
    # 分析CTR任务的特征重要性
    ctr_importance = analyze_feature_importance(
        df, 'ctr', processor.dense_features, processor.sparse_features
    )
    
    print("CTR任务 Top 10 重要特征:")
    for i, (feature, importance) in enumerate(ctr_importance['top_10_features'][:10]):
        print(f"  {i+1}. {feature}: {ctr_importance['feature_importance'][feature]:.4f}")
    
    # 3. 创建特征处理器
    print("\n3. 创建和拟合特征处理器...")
    feature_processor = create_dsp_feature_processor()
    dense_features, sparse_features = feature_processor.fit_transform(df)
    
    print(f"Dense特征维度: {dense_features.shape if dense_features is not None else 0}")
    print(f"Sparse特征维度: {sparse_features.shape}")
    
    # 显示嵌入规格
    embedding_specs = feature_processor.get_embedding_specs()
    print(f"总嵌入维度: {len(embedding_specs['vocab_sizes']) * embedding_specs['embedding_dim']}")
    
    # 显示高基数特征
    high_cardinality_features = [
        (k, v) for k, v in embedding_specs['vocab_sizes'].items() if v > 100
    ]
    if high_cardinality_features:
        print("高基数特征 (>100个唯一值):")
        for feature, cardinality in sorted(high_cardinality_features, key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {feature}: {cardinality}")
    
    # 4. 创建数据处理器和数据加载器
    print("\n4. 准备训练数据...")
    data_processor = DataProcessor(
        feature_processor=feature_processor,
        balance_strategy='weighted',
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    train_loader, val_loader, test_loader = data_processor.prepare_data(
        df=df,
        target_cols={'ctr': 'ctr', 'cvr': 'cvr', 'ivr': 'ivr'},
        scene_col='scene_id'
    )
    
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    
    # 5. 创建模型
    print("\n5. 创建多任务多场景模型...")
    dense_dim, sparse_dim = feature_processor.get_feature_dimensions()
    embedding_specs = feature_processor.get_embedding_specs()
    
    model = MultiTaskMultiSceneDSPModel(
        dense_dim=dense_dim,
        sparse_vocab_sizes=embedding_specs['vocab_sizes'],
        sparse_features=feature_processor.sparse_features,
        embedding_dim=8,
        num_scenes=3,
        num_experts=4,
        expert_hidden_dim=128,
        tower_hidden_dims=[128, 64],
        tasks=['ctr', 'cvr', 'ivr']
    )
    
    # 模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型参数:")
    print(f"  - 总参数: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    print(f"  - Dense特征维度: {dense_dim}")
    print(f"  - Sparse特征数量: {sparse_dim}")
    
    # 6. 简单的前向传播测试
    print("\n6. 测试模型前向传播...")
    model.eval()
    
    # 获取一个批次的数据
    sample_batch = next(iter(train_loader))
    dense_batch = sample_batch['dense_features']
    sparse_batch = sample_batch['sparse_features']
    scene_batch = sample_batch['scene_id']
    
    print(f"批次大小: {dense_batch.size(0)}")
    print(f"Dense特征形状: {dense_batch.shape}")
    print(f"Sparse特征形状: {sparse_batch.shape}")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(dense_batch, sparse_batch, scene_batch)
    
    print("模型输出:")
    for task, output in outputs.items():
        print(f"  {task}: 形状 {output.shape}, 范围 [{output.min():.4f}, {output.max():.4f}]")
    
    # 7. 快速训练示例
    print("\n7. 快速训练示例...")
    
    # 创建配置
    config = create_sample_config()
    config.training.num_epochs = 3  # 快速演示
    config.training.device = 'cpu'  # 确保使用CPU
    
    # 创建训练器
    trainer = DSPTrainer(
        model=model,
        device='cpu',
        log_dir='./example_logs',
        checkpoint_dir='./example_checkpoints'
    )
    
    try:
        # 训练
        print("开始训练...")
        train_history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=3,
            learning_rate=1e-3,
            patience=5
        )
        
        print("训练完成!")
        print(f"最佳验证损失: {train_history['best_val_loss']:.4f}")
        
        # 评估
        print("\n开始评估...")
        eval_results = trainer.evaluate(test_loader)
        
        print("评估结果:")
        for task, metrics in eval_results['overall_metrics'].items():
            print(f"  {task}:")
            for metric_name, value in metrics.items():
                print(f"    {metric_name}: {value:.4f}")
    
    finally:
        trainer.close()
    
    print("\n=== 示例完成 ===")

if __name__ == '__main__':
    main()