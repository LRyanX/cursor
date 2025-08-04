#!/usr/bin/env python3
"""
多任务多场景DSP广告模型主程序

该程序实现了一个完整的DSP广告预估系统，包括：
1. 多任务学习（CTR、CVR、IVR）
2. 多场景适配
3. 样本不均衡处理
4. 跷跷板问题解决
5. 指标校准
"""

import argparse
import os
import sys
import logging
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DSPConfig, create_sample_config, create_config_from_template, load_config
from multi_task_dsp_model import MultiTaskMultiSceneDSPModel
from data_utils import DataProcessor
from feature_engineering import create_dsp_feature_processor, create_sample_dsp_data
from trainer import DSPTrainer

def set_seed(seed: int):
    """设置随机种子确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_dir: str, level: str = 'INFO'):
    """设置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_level = getattr(logging, level.upper())
    
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 清除已有的handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 文件handler
    file_handler = logging.FileHandler(
        os.path.join(log_dir, 'main.log'), 
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_data(config: DSPConfig) -> pd.DataFrame:
    """加载数据"""
    if config.data.generate_sample_data:
        logging.info("生成DSP示例数据...")
        df = create_sample_dsp_data(
            num_samples=config.data.num_samples,
            num_scenes=config.model.num_scenes
        )
        logging.info(f"生成了 {len(df)} 个样本")
        
    elif config.data.data_path:
        logging.info(f"从文件加载数据: {config.data.data_path}")
        df = pd.read_csv(config.data.data_path)
        logging.info(f"加载了 {len(df)} 个样本")
    
    else:
        raise ValueError("必须指定数据路径或启用示例数据生成")
    
    return df

def create_model(config: DSPConfig, feature_processor) -> MultiTaskMultiSceneDSPModel:
    """创建模型"""
    logging.info("创建模型...")
    
    # 获取特征维度信息
    dense_dim, sparse_dim = feature_processor.get_feature_dimensions()
    embedding_specs = feature_processor.get_embedding_specs()
    
    model = MultiTaskMultiSceneDSPModel(
        dense_dim=dense_dim,
        sparse_vocab_sizes=embedding_specs['vocab_sizes'],
        sparse_features=feature_processor.sparse_features,
        embedding_dim=embedding_specs['embedding_dim'],
        num_scenes=config.model.num_scenes,
        num_experts=config.model.num_experts,
        expert_hidden_dim=config.model.expert_hidden_dim,
        tower_hidden_dims=config.model.tower_hidden_dims,
        tasks=config.model.tasks
    )
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"模型创建完成:")
    logging.info(f"  - 总参数数量: {total_params:,}")
    logging.info(f"  - 可训练参数: {trainable_params:,}")
    logging.info(f"  - 连续特征维度: {dense_dim}")
    logging.info(f"  - 离散特征数量: {sparse_dim}")
    logging.info(f"  - 任务数量: {len(config.model.tasks)}")
    logging.info(f"  - 场景数量: {config.model.num_scenes}")
    logging.info(f"  - 专家数量: {config.model.num_experts}")
    
    return model

def print_data_stats(df: pd.DataFrame, target_cols: dict, scene_col: str):
    """打印数据统计信息"""
    logging.info("=== 数据统计信息 ===")
    logging.info(f"数据集大小: {len(df):,}")
    logging.info(f"特征数量: {len([col for col in df.columns if col not in list(target_cols.values()) + [scene_col]])}")
    
    # 场景分布
    scene_counts = df[scene_col].value_counts().sort_index()
    logging.info(f"场景分布:")
    for scene_id, count in scene_counts.items():
        logging.info(f"  - 场景 {scene_id}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # 任务标签分布
    for task, col in target_cols.items():
        if col in df.columns:
            valid_mask = ~df[col].isna()
            if valid_mask.any():
                valid_data = df.loc[valid_mask, col]
                positive_ratio = valid_data.mean()
                logging.info(f"{task.upper()} 分布:")
                logging.info(f"  - 有效样本: {valid_mask.sum():,} ({valid_mask.mean()*100:.1f}%)")
                logging.info(f"  - 正样本比例: {positive_ratio:.4f}")
                if len(valid_data.unique()) == 2:
                    negative_count = (valid_data == 0).sum()
                    positive_count = (valid_data == 1).sum()
                    logging.info(f"  - 负样本: {negative_count:,}, 正样本: {positive_count:,}")

def main():
    parser = argparse.ArgumentParser(description='多任务多场景DSP广告模型训练')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--template', type=str, choices=['development', 'production', 'research'],
                       help='使用预定义配置模板')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'both'], default='both',
                       help='运行模式')
    parser.add_argument('--checkpoint', type=str, help='模型检查点路径（用于评估或继续训练）')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='输出目录')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='日志级别')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(str(output_dir / 'logs'), args.log_level)
    
    try:
        # 加载配置
        if args.config:
            config = load_config(args.config)
            logging.info(f"从文件加载配置: {args.config}")
        elif args.template:
            config = create_config_from_template(args.template)
            logging.info(f"使用模板配置: {args.template}")
        else:
            config = create_sample_config()
            logging.info("使用默认示例配置")
        
        # 验证配置
        errors = config.validate()
        if errors:
            logging.error("配置验证失败:")
            for error in errors:
                logging.error(f"  - {error}")
            return 1
        
        # 设置随机种子
        set_seed(config.experiment.seed)
        logging.info(f"设置随机种子: {config.experiment.seed}")
        
        # 更新输出路径
        config.training.log_dir = str(output_dir / 'logs')
        config.training.checkpoint_dir = str(output_dir / 'checkpoints')
        
        # 保存配置
        config.to_yaml(str(output_dir / 'config.yaml'))
        config.to_json(str(output_dir / 'config.json'))
        
        # 加载数据
        df = load_data(config)
        
        # 打印数据统计信息
        print_data_stats(df, config.data.target_cols, config.data.scene_col)
        
        # 特征处理
        logging.info("创建特征处理器...")
        feature_processor = create_dsp_feature_processor()
        
        # 数据预处理
        logging.info("准备数据...")
        data_processor = DataProcessor(
            feature_processor=feature_processor,
            balance_strategy=config.data.balance_strategy,
            test_size=config.data.test_size,
            val_size=config.data.val_size,
            random_state=config.data.random_state
        )
        
        train_loader, val_loader, test_loader = data_processor.prepare_data(
            df=df,
            target_cols=config.data.target_cols,
            scene_col=config.data.scene_col
        )
        
        # 打印特征统计信息
        feature_processor.print_feature_stats()
        
        logging.info(f"数据准备完成:")
        logging.info(f"  - 训练集批次数: {len(train_loader)}")
        logging.info(f"  - 验证集批次数: {len(val_loader)}")
        logging.info(f"  - 测试集批次数: {len(test_loader)}")
        
        # 创建模型
        model = create_model(config, feature_processor)
        
        # 创建训练器
        trainer = DSPTrainer(
            model=model,
            device=config.training.device,
            log_dir=config.training.log_dir,
            checkpoint_dir=config.training.checkpoint_dir
        )
        
        try:
            # 训练模式
            if args.mode in ['train', 'both']:
                logging.info("开始训练...")
                
                train_history = trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=config.training.num_epochs,
                    learning_rate=config.training.learning_rate,
                    weight_decay=config.training.weight_decay,
                    patience=config.training.patience,
                    scheduler_step_size=config.training.scheduler_step_size,
                    scheduler_gamma=config.training.scheduler_gamma
                )
                
                # 保存训练历史
                trainer.save_training_history(str(output_dir / 'training_history.json'))
                
                # 绘制训练曲线
                if config.experiment.plot_training_curves:
                    trainer.plot_training_curves(str(output_dir / 'training_curves.png'))
                
                logging.info("训练完成!")
            
            # 加载检查点（如果指定）
            if args.checkpoint:
                trainer._load_checkpoint(os.path.basename(args.checkpoint))
                logging.info(f"加载检查点: {args.checkpoint}")
            
            # 评估模式
            if args.mode in ['eval', 'both']:
                logging.info("开始评估...")
                
                eval_results = trainer.evaluate(test_loader)
                
                # 打印评估结果
                logging.info("=== 评估结果 ===")
                
                # 整体指标
                logging.info("整体指标:")
                for task, metrics in eval_results['overall_metrics'].items():
                    logging.info(f"  {task.upper()}:")
                    for metric_name, value in metrics.items():
                        logging.info(f"    - {metric_name}: {value:.4f}")
                
                # 场景指标
                if eval_results['scene_metrics']:
                    logging.info("场景指标:")
                    for scene, scene_metrics in eval_results['scene_metrics'].items():
                        logging.info(f"  {scene}:")
                        for task, task_metrics in scene_metrics.items():
                            logging.info(f"    {task.upper()}:")
                            for metric_name, value in task_metrics.items():
                                if metric_name in ['auc', 'pr_auc']:
                                    logging.info(f"      - {metric_name}: {value:.4f}")
                
                # 保存评估结果
                import json
                with open(output_dir / 'eval_results.json', 'w') as f:
                    # 转换numpy数组为列表以便JSON序列化
                    results_to_save = {
                        'overall_metrics': eval_results['overall_metrics'],
                        'scene_metrics': eval_results['scene_metrics']
                    }
                    json.dump(results_to_save, f, indent=2)
                
                logging.info("评估完成!")
        
        finally:
            trainer.close()
        
        logging.info(f"所有输出已保存到: {output_dir}")
        
        return 0
        
    except Exception as e:
        logging.error(f"程序执行失败: {str(e)}", exc_info=True)
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)