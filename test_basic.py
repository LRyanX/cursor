"""
基础测试脚本 - 验证代码结构和语法
"""
import os
import sys
import ast

def test_syntax():
    """测试所有Python文件的语法"""
    print("测试Python文件语法...")
    
    python_files = [
        'config.py',
        'utils.py', 
        'data_processor.py',
        'model.py',
        'main.py',
        'example.py'
    ]
    
    passed = 0
    total = len(python_files)
    
    for file in python_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 尝试解析AST
            ast.parse(content)
            print(f"✓ {file} 语法正确")
            passed += 1
            
        except SyntaxError as e:
            print(f"✗ {file} 语法错误: {e}")
        except FileNotFoundError:
            print(f"✗ {file} 文件不存在")
        except Exception as e:
            print(f"✗ {file} 其他错误: {e}")
    
    print(f"\n语法测试结果: {passed}/{total} 通过")
    return passed == total

def test_imports():
    """测试导入语句"""
    print("\n测试导入语句...")
    
    files_imports = {
        'config.py': ['dataclasses', 'typing', 'os'],
        'utils.py': ['numpy', 'pandas', 'torch', 'sklearn', 'matplotlib'],
        'data_processor.py': ['numpy', 'pandas', 'torch', 'sklearn'],
        'model.py': ['torch', 'numpy', 'typing'],
        'main.py': ['os', 'sys', 'argparse', 'logging', 'pathlib'],
        'example.py': ['os', 'sys', 'numpy', 'pandas', 'torch']
    }
    
    passed = 0
    total = len(files_imports)
    
    for file, expected_imports in files_imports.items():
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否包含预期的导入
            missing_imports = []
            for imp in expected_imports:
                if imp not in content:
                    missing_imports.append(imp)
            
            if not missing_imports:
                print(f"✓ {file} 导入语句正确")
                passed += 1
            else:
                print(f"✗ {file} 缺少导入: {missing_imports}")
                
        except Exception as e:
            print(f"✗ {file} 检查失败: {e}")
    
    print(f"\n导入测试结果: {passed}/{total} 通过")
    return passed == total

def test_classes():
    """测试类定义"""
    print("\n测试类定义...")
    
    expected_classes = {
        'config.py': ['ModelConfig', 'FeatureConfig', 'TrainingConfig'],
        'utils.py': ['FocalLoss', 'ParetoLoss', 'Calibrator'],
        'data_processor.py': ['FeatureProcessor', 'DSPDataset', 'DataLoader'],
        'model.py': ['ScenarioAdapter', 'TaskSpecificHead', 'MultiTaskModel', 'ParetoMultiTaskLoss', 'CalibratedModel', 'EnsembleModel', 'ModelTrainer']
    }
    
    passed = 0
    total = len(expected_classes)
    
    for file, expected_classes_list in expected_classes.items():
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否包含预期的类
            missing_classes = []
            for cls in expected_classes_list:
                if f"class {cls}" not in content:
                    missing_classes.append(cls)
            
            if not missing_classes:
                print(f"✓ {file} 类定义完整")
                passed += 1
            else:
                print(f"✗ {file} 缺少类: {missing_classes}")
                
        except Exception as e:
            print(f"✗ {file} 检查失败: {e}")
    
    print(f"\n类定义测试结果: {passed}/{total} 通过")
    return passed == total

def test_functions():
    """测试函数定义"""
    print("\n测试函数定义...")
    
    expected_functions = {
        'utils.py': ['set_seed', 'calculate_metrics', 'create_scenario_embeddings', 'balance_sample_weights'],
        'data_processor.py': ['create_sample_data'],
        'model.py': ['create_model', 'create_ensemble_models'],
        'main.py': ['setup_experiment', 'prepare_data', 'train_model', 'evaluate_model']
    }
    
    passed = 0
    total = len(expected_functions)
    
    for file, expected_functions_list in expected_functions.items():
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否包含预期的函数
            missing_functions = []
            for func in expected_functions_list:
                if f"def {func}" not in content:
                    missing_functions.append(func)
            
            if not missing_functions:
                print(f"✓ {file} 函数定义完整")
                passed += 1
            else:
                print(f"✗ {file} 缺少函数: {missing_functions}")
                
        except Exception as e:
            print(f"✗ {file} 检查失败: {e}")
    
    print(f"\n函数定义测试结果: {passed}/{total} 通过")
    return passed == total

def test_file_structure():
    """测试文件结构"""
    print("\n测试文件结构...")
    
    expected_files = [
        'config.py',
        'utils.py',
        'data_processor.py', 
        'model.py',
        'main.py',
        'example.py',
        'requirements.txt',
        'README.md'
    ]
    
    passed = 0
    total = len(expected_files)
    
    for file in expected_files:
        if os.path.exists(file):
            print(f"✓ {file} 存在")
            passed += 1
        else:
            print(f"✗ {file} 不存在")
    
    print(f"\n文件结构测试结果: {passed}/{total} 通过")
    return passed == total

def main():
    """主测试函数"""
    print("=" * 50)
    print("广告DSP多任务学习模型 - 基础测试")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_syntax,
        test_imports,
        test_classes,
        test_functions
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
        print("🎉 所有基础测试通过！")
        print("\n代码结构完整，包含以下功能:")
        print("✓ 配置文件 (config.py)")
        print("✓ 工具函数 (utils.py)")
        print("✓ 数据处理 (data_processor.py)")
        print("✓ 模型定义 (model.py)")
        print("✓ 主程序 (main.py)")
        print("✓ 示例脚本 (example.py)")
        print("✓ 依赖文件 (requirements.txt)")
        print("✓ 文档说明 (README.md)")
        print("\n要运行完整功能，请安装依赖包:")
        print("pip install -r requirements.txt")
    else:
        print("❌ 部分测试失败，请检查代码结构。")
    
    print("=" * 50)

if __name__ == "__main__":
    main()