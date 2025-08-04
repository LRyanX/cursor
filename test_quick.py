"""
快速测试脚本 - 验证所有模块是否正常工作
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有模块导入"""
    print("测试模块导入...")
    
    try:
        from config import model_config, feature_config, training_config
        print("✓ 配置模块导入成功")
    except Exception as e:
        print(f"✗ 配置模块导入失败: {e}")
        return False
    
    try:
        from utils import set_seed, FocalLoss, ParetoLoss, calculate_metrics
        print("✓ 工具模块导入成功")
    except Exception as e:
        print(f"✗ 工具模块导入失败: {e}")
        return False
    
    try:
        from data_processor import create_sample_data, DataLoader, FeatureProcessor
        print("✓ 数据处理模块导入成功")
    except Exception as e:
        print(f"✗ 数据处理模块导入失败: {e}")
        return False
    
    try:
        from model import create_model, MultiTaskModel, ModelTrainer
        print("✓ 模型模块导入成功")
    except Exception as e:
        print(f"✗ 模型模块导入失败: {e}")
        return False
    
    return True

def test_data_creation():
    """测试数据创建"""
    print("\n测试数据创建...")
    
    try:
        from data_processor import create_sample_data
        data = create_sample_data(n_samples=1000)
        print(f"✓ 示例数据创建成功，形状: {data.shape}")
        print(f"  CTR正样本比例: {data['ctr'].mean():.4f}")
        print(f"  CVR正样本比例: {data['cvr'].mean():.4f}")
        print(f"  IVR正样本比例: {data['ivr'].mean():.4f}")
        return True
    except Exception as e:
        print(f"✗ 数据创建失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n测试模型创建...")
    
    try:
        from config import model_config, feature_config
        from model import create_model
        
        model = create_model(model_config, feature_config)
        print(f"✓ 模型创建成功")
        
        # 测试前向传播
        batch_size = 32
        feature_dim = len(feature_config.sparse_features) + len(feature_config.numeric_features)
        scenario_dim = len(feature_config.scenario_features) * model_config.scenario_embedding_dim
        
        features = torch.randn(batch_size, feature_dim)
        scenario_features = torch.randn(batch_size, scenario_dim)
        
        predictions = model(features, scenario_features)
        print(f"✓ 前向传播成功，预测形状: {[pred.shape for pred in predictions.values()]}")
        
        return True
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return False

def test_training_components():
    """测试训练组件"""
    print("\n测试训练组件...")
    
    try:
        from config import model_config, feature_config, training_config
        from model import create_model, ModelTrainer
        
        model = create_model(model_config, feature_config)
        trainer = ModelTrainer(model, model_config, training_config)
        print("✓ 训练器创建成功")
        
        return True
    except Exception as e:
        print(f"✗ 训练器创建失败: {e}")
        return False

def test_loss_functions():
    """测试损失函数"""
    print("\n测试损失函数...")
    
    try:
        from utils import FocalLoss, ParetoLoss
        
        # 测试Focal Loss
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        inputs = torch.randn(10, 1)
        targets = torch.randint(0, 2, (10, 1)).float()
        loss = focal_loss(inputs, targets)
        print(f"✓ Focal Loss计算成功: {loss.item():.4f}")
        
        # 测试Pareto Loss
        pareto_loss = ParetoLoss(tasks=['ctr', 'cvr', 'ivr'])
        predictions = {
            'ctr': torch.randn(10, 1),
            'cvr': torch.randn(10, 1),
            'ivr': torch.randn(10, 1)
        }
        targets = {
            'ctr': torch.randint(0, 2, (10, 1)).float(),
            'cvr': torch.randint(0, 2, (10, 1)).float(),
            'ivr': torch.randint(0, 2, (10, 1)).float()
        }
        total_loss, task_losses = pareto_loss(predictions, targets)
        print(f"✓ Pareto Loss计算成功: {total_loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 损失函数测试失败: {e}")
        return False

def test_metrics():
    """测试评估指标"""
    print("\n测试评估指标...")
    
    try:
        from utils import calculate_metrics
        
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.random(100)
        
        metrics = calculate_metrics(y_true, y_pred, "test")
        print(f"✓ 评估指标计算成功")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 评估指标测试失败: {e}")
        return False

def test_data_processing():
    """测试数据处理"""
    print("\n测试数据处理...")
    
    try:
        from config import feature_config, training_config
        from data_processor import create_sample_data, DataLoader
        
        # 创建示例数据
        data = create_sample_data(n_samples=1000)
        data.to_csv("test_data.csv", index=False)
        
        # 测试数据加载器
        data_loader = DataLoader(feature_config, training_config)
        datasets, dataloaders = data_loader.load_and_preprocess("test_data.csv", ['ctr', 'cvr', 'ivr'])
        
        print(f"✓ 数据处理成功")
        print(f"  训练集大小: {len(datasets['train'])}")
        print(f"  验证集大小: {len(datasets['val'])}")
        print(f"  测试集大小: {len(datasets['test'])}")
        
        # 清理测试文件
        if os.path.exists("test_data.csv"):
            os.remove("test_data.csv")
        
        return True
    except Exception as e:
        print(f"✗ 数据处理测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("广告DSP多任务学习模型 - 快速测试")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_creation,
        test_model_creation,
        test_training_components,
        test_loss_functions,
        test_metrics,
        test_data_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！代码可以正常运行。")
        print("\n下一步:")
        print("1. 运行 python example.py 查看详细示例")
        print("2. 运行 python main.py --use_sample_data 进行完整训练")
    else:
        print("❌ 部分测试失败，请检查错误信息。")
    
    print("=" * 50)

if __name__ == "__main__":
    main()