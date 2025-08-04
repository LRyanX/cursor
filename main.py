"""
广告DSP多任务学习模型主程序
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import model_config, feature_config, training_config
from utils import set_seed, plot_calibration_curves, plot_training_curves
from data_processor import DataLoader, create_sample_data
from model import create_model, create_ensemble_models, ModelTrainer, CalibratedModel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_experiment(args):
    """设置实验环境"""
    # 设置随机种子
    set_seed(model_config.seed)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建模型保存目录
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    
    # 创建结果保存目录
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    return output_dir, model_dir, results_dir

def prepare_data(args):
    """准备数据"""
    logger.info("准备数据...")
    
    # 创建数据加载器
    data_loader = DataLoader(feature_config, training_config)
    
    if args.use_sample_data:
        # 使用示例数据
        logger.info("使用示例数据...")
        data = create_sample_data(n_samples=args.sample_size)
        data_path = "sample_data.csv"
        data.to_csv(data_path, index=False)
    else:
        # 使用真实数据
        data_path = args.data_path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    # 目标列
    target_columns = ['ctr', 'cvr', 'ivr']
    
    # 加载和预处理数据
    datasets, dataloaders = data_loader.load_and_preprocess(data_path, target_columns)
    
    # 保存特征处理器
    data_loader.save_processor(str(output_dir / "feature_processor.pkl"))
    
    logger.info(f"数据加载完成: 训练集 {len(datasets['train'])} 样本, "
                f"验证集 {len(datasets['val'])} 样本, "
                f"测试集 {len(datasets['test'])} 样本")
    
    return datasets, dataloaders, data_loader

def train_model(args, dataloaders, model_dir):
    """训练模型"""
    logger.info("开始训练模型...")
    
    # 创建模型
    model = create_model(model_config, feature_config)
    model = model.to(model_config.device)
    
    # 创建训练器
    trainer = ModelTrainer(model, model_config, training_config)
    
    # 训练模型
    train_losses, val_losses, val_metrics = trainer.train(
        dataloaders['train'], 
        dataloaders['val'],
        save_path=str(model_dir / "best_model.pth")
    )
    
    # 保存训练历史
    import joblib
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_metrics': val_metrics
    }
    joblib.dump(training_history, str(model_dir / "training_history.pkl"))
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, 
                        save_path=str(results_dir / "training_curves.png"))
    
    logger.info("模型训练完成")
    return model, trainer

def train_ensemble(args, dataloaders, model_dir):
    """训练集成模型"""
    logger.info("开始训练集成模型...")
    
    models = []
    for i in range(args.num_models):
        logger.info(f"训练模型 {i+1}/{args.num_models}")
        
        # 创建模型
        model = create_model(model_config, feature_config)
        model = model.to(model_config.device)
        
        # 创建训练器
        trainer = ModelTrainer(model, model_config, training_config)
        
        # 训练模型
        train_losses, val_losses, val_metrics = trainer.train(
            dataloaders['train'], 
            dataloaders['val'],
            save_path=str(model_dir / f"ensemble_model_{i}.pth")
        )
        
        models.append(model)
    
    # 创建集成模型
    from model import EnsembleModel
    ensemble_model = EnsembleModel(models)
    ensemble_model = ensemble_model.to(model_config.device)
    
    logger.info("集成模型训练完成")
    return ensemble_model

def evaluate_model(model, dataloaders, results_dir):
    """评估模型"""
    logger.info("开始评估模型...")
    
    model.eval()
    device = torch.device(model_config.device)
    
    # 测试集评估
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
    
    # 计算评估指标
    from utils import calculate_metrics
    test_metrics = {}
    for task in model_config.tasks:
        task_metrics = calculate_metrics(
            np.array(test_targets[task]),
            np.array(test_predictions[task]),
            task
        )
        test_metrics.update(task_metrics)
    
    # 保存结果
    results_df = pd.DataFrame([test_metrics])
    results_df.to_csv(str(results_dir / "test_metrics.csv"), index=False)
    
    # 绘制校准曲线
    plot_calibration_curves(
        test_targets, test_predictions,
        save_path=str(results_dir / "calibration_curves.png")
    )
    
    # 打印结果
    logger.info("测试集评估结果:")
    for metric, value in test_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    return test_metrics, test_predictions, test_targets

def calibrate_model(model, dataloaders, results_dir):
    """校准模型"""
    logger.info("开始校准模型...")
    
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
    calibrated_model = CalibratedModel(model, calibration_method=model_config.calibration_method)
    calibrated_model.fit_calibrators(val_predictions, val_targets)
    
    # 保存校准模型
    torch.save(calibrated_model.state_dict(), str(results_dir / "calibrated_model.pth"))
    
    logger.info("模型校准完成")
    return calibrated_model

def feature_importance_analysis(model, dataloaders, results_dir):
    """特征重要性分析"""
    logger.info("开始特征重要性分析...")
    
    # 获取测试集数据
    test_features = []
    test_targets = []
    
    for batch in dataloaders['test']:
        test_features.append(batch['features'].cpu().numpy())
        test_targets.append(batch['targets']['ctr'].cpu().numpy())
    
    X_test = np.concatenate(test_features, axis=0)
    y_test = np.concatenate(test_targets, axis=0)
    
    # 计算特征重要性
    from utils import feature_importance_analysis
    importance_scores = feature_importance_analysis(model, feature_config.feature_names, X_test, y_test)
    
    # 保存特征重要性
    importance_df = pd.DataFrame(list(importance_scores.items()), columns=['feature', 'importance'])
    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df.to_csv(str(results_dir / "feature_importance.csv"), index=False)
    
    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig(str(results_dir / "feature_importance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("特征重要性分析完成")

def hyperparameter_optimization(args, dataloaders, model_dir):
    """超参数优化"""
    if not args.optimize_hyperparams:
        return
    
    logger.info("开始超参数优化...")
    
    import optuna
    
    def objective(trial):
        # 超参数搜索空间
        lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        
        # 更新配置
        model_config.learning_rate = lr
        model_config.hidden_dims = [hidden_dim, hidden_dim // 2]
        model_config.dropout_rate = dropout_rate
        
        # 创建模型
        model = create_model(model_config, feature_config)
        model = model.to(model_config.device)
        
        # 创建训练器
        trainer = ModelTrainer(model, model_config, training_config)
        
        # 训练（减少epoch数用于快速搜索）
        original_epochs = model_config.num_epochs
        model_config.num_epochs = 10
        
        train_losses, val_losses, val_metrics = trainer.train(
            dataloaders['train'], dataloaders['val']
        )
        
        model_config.num_epochs = original_epochs
        
        # 返回验证损失作为优化目标
        return sum(val_losses.values())
    
    # 创建研究
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.n_trials)
    
    # 保存最佳参数
    best_params = study.best_params
    import json
    with open(str(results_dir / "best_hyperparams.json"), 'w') as f:
        json.dump(best_params, f, indent=2)
    
    logger.info(f"超参数优化完成，最佳参数: {best_params}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='广告DSP多任务学习模型')
    parser.add_argument('--data_path', type=str, default='', help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--use_sample_data', action='store_true', help='使用示例数据')
    parser.add_argument('--sample_size', type=int, default=10000, help='示例数据大小')
    parser.add_argument('--ensemble', action='store_true', help='使用集成模型')
    parser.add_argument('--num_models', type=int, default=3, help='集成模型数量')
    parser.add_argument('--optimize_hyperparams', action='store_true', help='进行超参数优化')
    parser.add_argument('--n_trials', type=int, default=20, help='超参数优化试验次数')
    parser.add_argument('--calibrate', action='store_true', help='校准模型')
    parser.add_argument('--feature_importance', action='store_true', help='分析特征重要性')
    
    args = parser.parse_args()
    
    # 设置实验环境
    global output_dir, model_dir, results_dir
    output_dir, model_dir, results_dir = setup_experiment(args)
    
    try:
        # 准备数据
        datasets, dataloaders, data_loader = prepare_data(args)
        
        # 超参数优化
        hyperparameter_optimization(args, dataloaders, model_dir)
        
        # 训练模型
        if args.ensemble:
            model = train_ensemble(args, dataloaders, model_dir)
        else:
            model, trainer = train_model(args, dataloaders, model_dir)
        
        # 校准模型
        if args.calibrate:
            model = calibrate_model(model, dataloaders, results_dir)
        
        # 评估模型
        test_metrics, test_predictions, test_targets = evaluate_model(model, dataloaders, results_dir)
        
        # 特征重要性分析
        if args.feature_importance:
            feature_importance_analysis(model, dataloaders, results_dir)
        
        logger.info("实验完成！")
        
    except Exception as e:
        logger.error(f"实验失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()