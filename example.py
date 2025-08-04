"""
广告DSP多任务学习模型示例脚本
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import model_config, feature_config, training_config
from utils import set_seed, plot_calibration_curves
from data_processor import create_sample_data, DataLoader
from model import create_model, ModelTrainer

def run_basic_example():
    """运行基础示例"""
    print("=" * 50)
    print("广告DSP多任务学习模型示例")
    print("=" * 50)
    
    # 设置随机种子
    set_seed(42)
    
    # 1. 创建示例数据
    print("\n1. 创建示例数据...")
    data = create_sample_data(n_samples=5000)
    print(f"数据形状: {data.shape}")
    print(f"CTR正样本比例: {data['ctr'].mean():.4f}")
    print(f"CVR正样本比例: {data['cvr'].mean():.4f}")
    print(f"IVR正样本比例: {data['ivr'].mean():.4f}")
    
    # 2. 数据预处理
    print("\n2. 数据预处理...")
    data_loader = DataLoader(feature_config, training_config)
    datasets, dataloaders = data_loader.load_and_preprocess("sample_data.csv", ['ctr', 'cvr', 'ivr'])
    
    # 3. 创建模型
    print("\n3. 创建模型...")
    model = create_model(model_config, feature_config)
    model = model.to(model_config.device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 4. 训练模型
    print("\n4. 训练模型...")
    trainer = ModelTrainer(model, model_config, training_config)
    
    # 减少训练轮数用于演示
    original_epochs = model_config.num_epochs
    model_config.num_epochs = 20
    
    train_losses, val_losses, val_metrics = trainer.train(
        dataloaders['train'], 
        dataloaders['val']
    )
    
    model_config.num_epochs = original_epochs
    
    # 5. 评估模型
    print("\n5. 评估模型...")
    model.eval()
    device = torch.device(model_config.device)
    
    test_predictions = {task: [] for task in model_config.tasks}
    test_targets = {task: [] for task in model_config.tasks}
    
    with torch.no_grad():
        for batch in dataloaders['test']:
            features = batch['features'].to(device)
            targets = {task: target.to(device) for task, target in batch['targets'].items()}
            
            predictions = model(features, features)
            
            for task in model_config.tasks:
                test_predictions[task].extend(torch.sigmoid(predictions[task]).cpu().numpy())
                test_targets[task].extend(targets[task].cpu().numpy())
    
    # 6. 计算评估指标
    from utils import calculate_metrics
    print("\n6. 评估结果:")
    for task in model_config.tasks:
        task_metrics = calculate_metrics(
            np.array(test_targets[task]),
            np.array(test_predictions[task]),
            task
        )
        print(f"\n{task.upper()} 任务:")
        for metric, value in task_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # 7. 绘制校准曲线
    print("\n7. 生成校准曲线...")
    plot_calibration_curves(test_targets, test_predictions)
    
    print("\n示例运行完成！")

def run_advanced_example():
    """运行高级示例"""
    print("=" * 50)
    print("高级功能示例")
    print("=" * 50)
    
    # 设置随机种子
    set_seed(42)
    
    # 1. 创建更大规模的数据
    print("\n1. 创建大规模示例数据...")
    data = create_sample_data(n_samples=20000)
    data.to_csv("large_sample_data.csv", index=False)
    
    # 2. 数据预处理
    print("\n2. 数据预处理...")
    data_loader = DataLoader(feature_config, training_config)
    datasets, dataloaders = data_loader.load_and_preprocess("large_sample_data.csv", ['ctr', 'cvr', 'ivr'])
    
    # 3. 创建集成模型
    print("\n3. 创建集成模型...")
    from model import create_ensemble_models, EnsembleModel
    
    models = create_ensemble_models(model_config, feature_config, num_models=3)
    ensemble_model = EnsembleModel(models)
    ensemble_model = ensemble_model.to(model_config.device)
    
    # 4. 训练集成模型
    print("\n4. 训练集成模型...")
    for i, model in enumerate(models):
        print(f"训练模型 {i+1}/3...")
        trainer = ModelTrainer(model, model_config, training_config)
        
        # 减少训练轮数
        original_epochs = model_config.num_epochs
        model_config.num_epochs = 15
        
        trainer.train(dataloaders['train'], dataloaders['val'])
        
        model_config.num_epochs = original_epochs
    
    # 5. 评估集成模型
    print("\n5. 评估集成模型...")
    ensemble_model.eval()
    device = torch.device(model_config.device)
    
    ensemble_predictions = {task: [] for task in model_config.tasks}
    test_targets = {task: [] for task in model_config.tasks}
    
    with torch.no_grad():
        for batch in dataloaders['test']:
            features = batch['features'].to(device)
            targets = {task: target.to(device) for task, target in batch['targets'].items()}
            
            predictions = ensemble_model(features, features)
            
            for task in model_config.tasks:
                ensemble_predictions[task].extend(torch.sigmoid(predictions[task]).cpu().numpy())
                test_targets[task].extend(targets[task].cpu().numpy())
    
    # 6. 比较集成模型和单模型性能
    print("\n6. 集成模型评估结果:")
    from utils import calculate_metrics
    for task in model_config.tasks:
        task_metrics = calculate_metrics(
            np.array(test_targets[task]),
            np.array(ensemble_predictions[task]),
            task
        )
        print(f"\n{task.upper()} 任务 (集成模型):")
        for metric, value in task_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # 7. 特征重要性分析
    print("\n7. 特征重要性分析...")
    from utils import feature_importance_analysis
    
    # 获取测试集数据
    test_features = []
    test_targets_ctr = []
    
    for batch in dataloaders['test']:
        test_features.append(batch['features'].cpu().numpy())
        test_targets_ctr.append(batch['targets']['ctr'].cpu().numpy())
    
    X_test = np.concatenate(test_features, axis=0)
    y_test = np.concatenate(test_targets_ctr, axis=0)
    
    # 计算特征重要性
    importance_scores = feature_importance_analysis(ensemble_model, feature_config.feature_names, X_test, y_test)
    
    # 显示前10个重要特征
    sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    print("\n前10个重要特征:")
    for i, (feature, importance) in enumerate(sorted_importance[:10]):
        print(f"  {i+1}. {feature}: {importance:.4f}")
    
    print("\n高级示例运行完成！")

def demonstrate_calibration():
    """演示校准功能"""
    print("=" * 50)
    print("模型校准演示")
    print("=" * 50)
    
    # 设置随机种子
    set_seed(42)
    
    # 1. 创建数据
    print("\n1. 创建数据...")
    data = create_sample_data(n_samples=10000)
    data.to_csv("calibration_data.csv", index=False)
    
    # 2. 数据预处理
    print("\n2. 数据预处理...")
    data_loader = DataLoader(feature_config, training_config)
    datasets, dataloaders = data_loader.load_and_preprocess("calibration_data.csv", ['ctr', 'cvr', 'ivr'])
    
    # 3. 训练模型
    print("\n3. 训练模型...")
    model = create_model(model_config, feature_config)
    model = model.to(model_config.device)
    
    trainer = ModelTrainer(model, model_config, training_config)
    
    # 减少训练轮数
    original_epochs = model_config.num_epochs
    model_config.num_epochs = 25
    
    trainer.train(dataloaders['train'], dataloaders['val'])
    
    model_config.num_epochs = original_epochs
    
    # 4. 校准模型
    print("\n4. 校准模型...")
    from model import CalibratedModel
    
    # 获取验证集预测用于校准
    model.eval()
    device = torch.device(model_config.device)
    
    val_predictions = {task: [] for task in model_config.tasks}
    val_targets = {task: [] for task in model_config.tasks}
    
    with torch.no_grad():
        for batch in dataloaders['val']:
            features = batch['features'].to(device)
            targets = {task: target.to(device) for task, target in batch['targets'].items()}
            
            predictions = model(features, features)
            
            for task in model_config.tasks:
                val_predictions[task].extend(torch.sigmoid(predictions[task]).cpu().numpy())
                val_targets[task].extend(targets[task].cpu().numpy())
    
    # 创建校准模型
    calibrated_model = CalibratedModel(model, calibration_method="isotonic")
    calibrated_model.fit_calibrators(val_predictions, val_targets)
    
    # 5. 比较校准前后的效果
    print("\n5. 校准效果比较...")
    
    # 测试集评估
    test_predictions_raw = {task: [] for task in model_config.tasks}
    test_predictions_calibrated = {task: [] for task in model_config.tasks}
    test_targets = {task: [] for task in model_config.tasks}
    
    with torch.no_grad():
        for batch in dataloaders['test']:
            features = batch['features'].to(device)
            targets = {task: target.to(device) for task, target in batch['targets'].items()}
            
            # 原始预测
            raw_predictions = model(features, features)
            
            # 校准预测
            calibrated_predictions = calibrated_model(features, features)
            
            for task in model_config.tasks:
                test_predictions_raw[task].extend(torch.sigmoid(raw_predictions[task]).cpu().numpy())
                test_predictions_calibrated[task].extend(calibrated_predictions[task].cpu().numpy())
                test_targets[task].extend(targets[task].cpu().numpy())
    
    # 计算校准误差
    from utils import calculate_metrics
    print("\n校准前后对比:")
    for task in model_config.tasks:
        raw_metrics = calculate_metrics(
            np.array(test_targets[task]),
            np.array(test_predictions_raw[task]),
            task
        )
        calibrated_metrics = calculate_metrics(
            np.array(test_targets[task]),
            np.array(test_predictions_calibrated[task]),
            task
        )
        
        print(f"\n{task.upper()} 任务:")
        print(f"  校准前校准误差: {raw_metrics[f'{task}_calibration_error']:.4f}")
        print(f"  校准后校准误差: {calibrated_metrics[f'{task}_calibration_error']:.4f}")
        print(f"  改进: {raw_metrics[f'{task}_calibration_error'] - calibrated_metrics[f'{task}_calibration_error']:.4f}")
    
    print("\n校准演示完成！")

if __name__ == "__main__":
    print("选择要运行的示例:")
    print("1. 基础示例")
    print("2. 高级示例 (集成模型)")
    print("3. 校准演示")
    print("4. 运行所有示例")
    
    choice = input("\n请输入选择 (1-4): ").strip()
    
    if choice == "1":
        run_basic_example()
    elif choice == "2":
        run_advanced_example()
    elif choice == "3":
        demonstrate_calibration()
    elif choice == "4":
        run_basic_example()
        print("\n" + "="*50 + "\n")
        run_advanced_example()
        print("\n" + "="*50 + "\n")
        demonstrate_calibration()
    else:
        print("无效选择，运行基础示例...")
        run_basic_example()