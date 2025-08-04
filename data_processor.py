"""
数据处理模块
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FeatureProcessor:
    """特征处理器"""
    def __init__(self, feature_config):
        self.feature_config = feature_config
        self.label_encoders = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_names = []
        
    def fit_transform(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """拟合并转换特征"""
        processed_features = []
        feature_names = []
        
        # 处理数值特征
        if self.feature_config.numeric_features:
            numeric_data = data[self.feature_config.numeric_features].fillna(0)
            scaler = StandardScaler()
            scaled_numeric = scaler.fit_transform(numeric_data)
            self.scalers['numeric'] = scaler
            processed_features.append(scaled_numeric)
            feature_names.extend(self.feature_config.numeric_features)
        
        # 处理类别特征
        if self.feature_config.categorical_features:
            categorical_data = data[self.feature_config.categorical_features].fillna('unknown')
            encoded_features = []
            
            for feature in self.feature_config.categorical_features:
                if feature in data.columns:
                    le = LabelEncoder()
                    encoded_feature = le.fit_transform(categorical_data[feature])
                    self.label_encoders[feature] = le
                    encoded_features.append(encoded_feature.reshape(-1, 1))
                    feature_names.append(feature)
            
            if encoded_features:
                categorical_encoded = np.concatenate(encoded_features, axis=1)
                processed_features.append(categorical_encoded)
        
        # 处理稀疏特征（除了已处理的类别特征）
        sparse_features = [f for f in self.feature_config.sparse_features 
                         if f not in self.feature_config.categorical_features 
                         and f not in self.feature_config.numeric_features]
        
        if sparse_features:
            sparse_data = data[sparse_features].fillna('unknown')
            sparse_encoded_features = []
            
            for feature in sparse_features:
                if feature in data.columns:
                    le = LabelEncoder()
                    encoded_feature = le.fit_transform(sparse_data[feature])
                    self.label_encoders[feature] = le
                    sparse_encoded_features.append(encoded_feature.reshape(-1, 1))
                    feature_names.append(feature)
            
            if sparse_encoded_features:
                sparse_encoded = np.concatenate(sparse_encoded_features, axis=1)
                processed_features.append(sparse_encoded)
        
        # 合并所有特征
        if processed_features:
            X = np.concatenate(processed_features, axis=1)
        else:
            X = np.zeros((len(data), 1))
        
        self.feature_names = feature_names
        return X, feature_names
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """转换特征（用于测试数据）"""
        processed_features = []
        
        # 处理数值特征
        if self.feature_config.numeric_features:
            numeric_data = data[self.feature_config.numeric_features].fillna(0)
            scaled_numeric = self.scalers['numeric'].transform(numeric_data)
            processed_features.append(scaled_numeric)
        
        # 处理类别特征
        if self.feature_config.categorical_features:
            categorical_data = data[self.feature_config.categorical_features].fillna('unknown')
            encoded_features = []
            
            for feature in self.feature_config.categorical_features:
                if feature in data.columns:
                    le = self.label_encoders[feature]
                    # 处理未见过的类别
                    unique_values = le.classes_
                    encoded_feature = []
                    for val in categorical_data[feature]:
                        if val in unique_values:
                            encoded_feature.append(le.transform([val])[0])
                        else:
                            encoded_feature.append(0)  # 默认值
                    encoded_features.append(np.array(encoded_feature).reshape(-1, 1))
            
            if encoded_features:
                categorical_encoded = np.concatenate(encoded_features, axis=1)
                processed_features.append(categorical_encoded)
        
        # 处理稀疏特征
        sparse_features = [f for f in self.feature_config.sparse_features 
                         if f not in self.feature_config.categorical_features 
                         and f not in self.feature_config.numeric_features]
        
        if sparse_features:
            sparse_data = data[sparse_features].fillna('unknown')
            sparse_encoded_features = []
            
            for feature in sparse_features:
                if feature in data.columns:
                    le = self.label_encoders[feature]
                    unique_values = le.classes_
                    encoded_feature = []
                    for val in sparse_data[feature]:
                        if val in unique_values:
                            encoded_feature.append(le.transform([val])[0])
                        else:
                            encoded_feature.append(0)
                    sparse_encoded_features.append(np.array(encoded_feature).reshape(-1, 1))
            
            if sparse_encoded_features:
                sparse_encoded = np.concatenate(sparse_encoded_features, axis=1)
                processed_features.append(sparse_encoded)
        
        # 合并所有特征
        if processed_features:
            X = np.concatenate(processed_features, axis=1)
        else:
            X = np.zeros((len(data), 1))
        
        return X
    
    def feature_selection(self, X: np.ndarray, y: np.ndarray, k: int = 100) -> np.ndarray:
        """特征选择"""
        if X.shape[1] <= k:
            return X
        
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        self.feature_selectors['main'] = selector
        
        # 更新特征名称
        selected_indices = selector.get_support()
        self.feature_names = [self.feature_names[i] for i in range(len(self.feature_names)) if selected_indices[i]]
        
        return X_selected
    
    def save(self, path: str):
        """保存处理器"""
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scalers': self.scalers,
            'feature_selectors': self.feature_selectors,
            'feature_names': self.feature_names
        }, path)
    
    def load(self, path: str):
        """加载处理器"""
        saved_data = joblib.load(path)
        self.label_encoders = saved_data['label_encoders']
        self.scalers = saved_data['scalers']
        self.feature_selectors = saved_data['feature_selectors']
        self.feature_names = saved_data['feature_names']

class DSPDataset(Dataset):
    """DSP数据集"""
    def __init__(self, X: np.ndarray, targets: Dict[str, np.ndarray], 
                 sample_weights: Optional[np.ndarray] = None):
        self.X = torch.FloatTensor(X)
        self.targets = {task: torch.FloatTensor(target) for task, target in targets.items()}
        self.sample_weights = torch.FloatTensor(sample_weights) if sample_weights is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        item = {
            'features': self.X[idx],
            'targets': {task: target[idx] for task, target in self.targets.items()}
        }
        if self.sample_weights is not None:
            item['sample_weight'] = self.sample_weights[idx]
        return item

class DataLoader:
    """数据加载器"""
    def __init__(self, feature_config, training_config):
        self.feature_config = feature_config
        self.training_config = training_config
        self.feature_processor = FeatureProcessor(feature_config)
        
    def load_and_preprocess(self, data_path: str, target_columns: List[str]) -> Tuple[Dict, Dict]:
        """加载并预处理数据"""
        # 加载数据
        data = pd.read_csv(data_path)
        
        # 分离特征和目标
        feature_data = data.drop(columns=target_columns, errors='ignore')
        target_data = data[target_columns]
        
        # 处理特征
        X, feature_names = self.feature_processor.fit_transform(feature_data)
        
        # 创建场景嵌入
        from utils import create_scenario_embeddings
        scenario_vector, scenario_embeddings = create_scenario_embeddings(
            data, self.feature_config.scenario_features
        )
        
        # 合并特征和场景向量
        if scenario_vector.shape[1] > 0:
            X = np.concatenate([X, scenario_vector], axis=1)
            feature_names.extend([f'scenario_{i}' for i in range(scenario_vector.shape[1])])
        
        # 特征选择
        if self.feature_config.feature_selection and len(target_columns) > 0:
            # 使用第一个目标进行特征选择
            X = self.feature_processor.feature_selection(X, target_data[target_columns[0]])
        
        # 分割数据
        train_data, temp_data = train_test_split(
            data, test_size=1-self.training_config.train_ratio, random_state=42
        )
        
        val_data, test_data = train_test_split(
            temp_data, 
            test_size=self.training_config.test_ratio/(self.training_config.val_ratio + self.training_config.test_ratio),
            random_state=42
        )
        
        # 处理训练数据
        train_X, _ = self.feature_processor.fit_transform(train_data.drop(columns=target_columns, errors='ignore'))
        train_scenario_vector, _ = create_scenario_embeddings(
            train_data, self.feature_config.scenario_features
        )
        if train_scenario_vector.shape[1] > 0:
            train_X = np.concatenate([train_X, train_scenario_vector], axis=1)
        
        # 处理验证数据
        val_X = self.feature_processor.transform(val_data.drop(columns=target_columns, errors='ignore'))
        val_scenario_vector, _ = create_scenario_embeddings(
            val_data, self.feature_config.scenario_features
        )
        if val_scenario_vector.shape[1] > 0:
            val_X = np.concatenate([val_X, val_scenario_vector], axis=1)
        
        # 处理测试数据
        test_X = self.feature_processor.transform(test_data.drop(columns=target_columns, errors='ignore'))
        test_scenario_vector, _ = create_scenario_embeddings(
            test_data, self.feature_config.scenario_features
        )
        if test_scenario_vector.shape[1] > 0:
            test_X = np.concatenate([test_X, test_scenario_vector], axis=1)
        
        # 创建目标字典
        train_targets = {col: train_data[col].values for col in target_columns}
        val_targets = {col: val_data[col].values for col in target_columns}
        test_targets = {col: test_data[col].values for col in target_columns}
        
        # 计算样本权重
        from utils import balance_sample_weights
        train_weights = {}
        for task in target_columns:
            train_weights[task] = balance_sample_weights(
                train_targets[task], task, self.training_config.sample_weights
            )
        
        # 创建数据集
        datasets = {
            'train': DSPDataset(train_X, train_targets, 
                              sample_weights=np.mean(list(train_weights.values()), axis=0)),
            'val': DSPDataset(val_X, val_targets),
            'test': DSPDataset(test_X, test_targets)
        }
        
        # 创建数据加载器
        dataloaders = {
            'train': torch.utils.data.DataLoader(
                datasets['train'], 
                batch_size=self.training_config.batch_size, 
                shuffle=True, 
                num_workers=4
            ),
            'val': torch.utils.data.DataLoader(
                datasets['val'], 
                batch_size=self.training_config.batch_size, 
                shuffle=False, 
                num_workers=4
            ),
            'test': torch.utils.data.DataLoader(
                datasets['test'], 
                batch_size=self.training_config.batch_size, 
                shuffle=False, 
                num_workers=4
            )
        }
        
        return datasets, dataloaders
    
    def save_processor(self, path: str):
        """保存特征处理器"""
        self.feature_processor.save(path)
    
    def load_processor(self, path: str):
        """加载特征处理器"""
        self.feature_processor.load(path)

def create_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """创建示例数据"""
    np.random.seed(42)
    
    # 创建稀疏特征
    sparse_features = [
        'hour', 'weekday', 'adv_id', 'affiliate_id', 'campaign_id', 'ad_group_id', 
        'ad_id', 'creative_id', 'feature_1', 'pos', 'instl', 'response_type', 
        'ad_format', 'os', 'device_make', 'bundle_id', 'country', 'package', 
        'category', 'connection_type', 'device_model', 'lang', 'publisher_id', 
        'first_ssp', 'last_ssp', 'video_placement', 'is_rewarded', 'offer_id',
        'supply_developer_id', 'supply_genreId', 'supply_version', 
        'supply_minimum_os_version', 'supply_industry_id', 'is_oem', 'tag_id', 
        'osv', 'ua', 'demand_developer_id', 'demand_genreId', 'demand_version',
        'demand_minimum_os_version', 'demand_industry_id', 'ip', 'device_id'
    ]
    
    data = {}
    
    # 数值特征
    data['ad_width'] = np.random.randint(300, 1200, n_samples)
    data['ad_height'] = np.random.randint(250, 800, n_samples)
    data['hour'] = np.random.randint(0, 24, n_samples)
    data['weekday'] = np.random.randint(0, 7, n_samples)
    
    # 稀疏特征
    for feature in sparse_features:
        if feature in ['ad_width', 'ad_height', 'hour', 'weekday']:
            continue
        
        if feature.startswith('supply_') or feature == 'ip':
            # 场景特征，减少类别数
            n_categories = np.random.randint(5, 20)
        else:
            n_categories = np.random.randint(10, 100)
        
        categories = [f'{feature}_{i}' for i in range(n_categories)]
        data[feature] = np.random.choice(categories, n_samples)
    
    # 创建目标变量（模拟真实场景的稀疏性）
    # CTR: 点击率 ~1%
    data['ctr'] = np.random.binomial(1, 0.01, n_samples)
    
    # CVR: 转化率 ~0.1% (在点击的基础上)
    data['cvr'] = np.zeros(n_samples)
    click_indices = np.where(data['ctr'] == 1)[0]
    data['cvr'][click_indices] = np.random.binomial(1, 0.1, len(click_indices))
    
    # IVR: 曝光转化率 ~0.001%
    data['ivr'] = np.random.binomial(1, 0.0001, n_samples)
    
    return pd.DataFrame(data)