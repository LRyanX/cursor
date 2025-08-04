import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import logging

class FeatureProcessor:
    """特征处理器，处理dense和sparse特征"""
    
    def __init__(self, 
                 dense_features: List[str],
                 sparse_features: List[str],
                 embedding_dim: int = 8,
                 normalize_dense: bool = True,
                 handle_unknown: str = 'ignore'):
        """
        Args:
            dense_features: 连续特征列表
            sparse_features: 离散特征列表
            embedding_dim: 嵌入维度
            normalize_dense: 是否标准化连续特征
            handle_unknown: 处理未知类别的策略
        """
        self.dense_features = dense_features
        self.sparse_features = sparse_features
        self.embedding_dim = embedding_dim
        self.normalize_dense = normalize_dense
        self.handle_unknown = handle_unknown
        
        # 特征处理器
        self.dense_scaler = StandardScaler() if normalize_dense else None
        self.sparse_encoders = {}
        self.sparse_vocab_sizes = {}
        
        # 特征统计
        self.feature_stats = {}
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame) -> 'FeatureProcessor':
        """拟合特征处理器"""
        logging.info("开始拟合特征处理器...")
        
        # 处理连续特征
        if self.dense_features and self.normalize_dense:
            dense_data = df[self.dense_features].fillna(0)
            self.dense_scaler.fit(dense_data)
            logging.info(f"拟合了 {len(self.dense_features)} 个连续特征的标准化器")
        
        # 处理离散特征
        for feature in self.sparse_features:
            if feature in df.columns:
                # 填充缺失值
                series = df[feature].fillna('unknown')
                
                # 创建标签编码器
                encoder = LabelEncoder()
                
                # 如果处理未知值，添加特殊标记
                if self.handle_unknown == 'ignore':
                    unique_values = list(series.unique()) + ['<UNK>']
                    encoder.fit(unique_values)
                else:
                    encoder.fit(series)
                
                self.sparse_encoders[feature] = encoder
                self.sparse_vocab_sizes[feature] = len(encoder.classes_)
                
                logging.info(f"特征 {feature}: {self.sparse_vocab_sizes[feature]} 个唯一值")
        
        # 计算特征统计
        self._compute_feature_stats(df)
        
        self.is_fitted = True
        logging.info("特征处理器拟合完成")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """转换特征"""
        if not self.is_fitted:
            raise ValueError("特征处理器尚未拟合，请先调用 fit() 方法")
        
        # 处理连续特征
        dense_data = None
        if self.dense_features:
            dense_data = df[self.dense_features].fillna(0).values.astype(np.float32)
            
            if self.normalize_dense and self.dense_scaler:
                dense_data = self.dense_scaler.transform(dense_data)
        
        # 处理离散特征
        sparse_data = []
        for feature in self.sparse_features:
            if feature in df.columns:
                series = df[feature].fillna('unknown')
                
                if feature in self.sparse_encoders:
                    encoder = self.sparse_encoders[feature]
                    
                    # 处理未知值
                    if self.handle_unknown == 'ignore':
                        encoded = []
                        for val in series:
                            try:
                                encoded.append(encoder.transform([val])[0])
                            except ValueError:
                                # 未知值映射到 <UNK>
                                encoded.append(encoder.transform(['<UNK>'])[0])
                        encoded = np.array(encoded)
                    else:
                        encoded = encoder.transform(series)
                    
                    sparse_data.append(encoded)
                else:
                    # 如果特征不在编码器中，用0填充
                    sparse_data.append(np.zeros(len(df), dtype=np.int64))
        
        sparse_data = np.column_stack(sparse_data) if sparse_data else np.zeros((len(df), 0), dtype=np.int64)
        
        return dense_data, sparse_data
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """拟合并转换特征"""
        return self.fit(df).transform(df)
    
    def _compute_feature_stats(self, df: pd.DataFrame):
        """计算特征统计信息"""
        self.feature_stats = {
            'dense_features': {},
            'sparse_features': {},
            'total_samples': len(df)
        }
        
        # 连续特征统计
        for feature in self.dense_features:
            if feature in df.columns:
                series = df[feature]
                self.feature_stats['dense_features'][feature] = {
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'missing_ratio': series.isna().mean()
                }
        
        # 离散特征统计
        for feature in self.sparse_features:
            if feature in df.columns:
                series = df[feature]
                self.feature_stats['sparse_features'][feature] = {
                    'unique_count': series.nunique(),
                    'missing_ratio': series.isna().mean(),
                    'top_values': series.value_counts().head(5).to_dict()
                }
    
    def get_feature_dimensions(self) -> Tuple[int, int]:
        """获取特征维度"""
        dense_dim = len(self.dense_features)
        sparse_dim = len(self.sparse_features)
        return dense_dim, sparse_dim
    
    def get_embedding_specs(self) -> Dict[str, int]:
        """获取嵌入规格"""
        return {
            'vocab_sizes': self.sparse_vocab_sizes,
            'embedding_dim': self.embedding_dim,
            'num_sparse_features': len(self.sparse_features)
        }
    
    def print_feature_stats(self):
        """打印特征统计信息"""
        if not self.feature_stats:
            logging.warning("特征统计信息不可用")
            return
        
        logging.info("=== 特征统计信息 ===")
        logging.info(f"总样本数: {self.feature_stats['total_samples']:,}")
        
        # 连续特征
        logging.info(f"\n连续特征 ({len(self.dense_features)} 个):")
        for feature, stats in self.feature_stats['dense_features'].items():
            logging.info(f"  {feature}:")
            logging.info(f"    均值: {stats['mean']:.4f}, 标准差: {stats['std']:.4f}")
            logging.info(f"    范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
            logging.info(f"    缺失率: {stats['missing_ratio']:.4f}")
        
        # 离散特征
        logging.info(f"\n离散特征 ({len(self.sparse_features)} 个):")
        for feature, stats in self.feature_stats['sparse_features'].items():
            logging.info(f"  {feature}:")
            logging.info(f"    唯一值数量: {stats['unique_count']}")
            logging.info(f"    缺失率: {stats['missing_ratio']:.4f}")
            if stats['top_values']:
                top_str = ", ".join([f"{k}:{v}" for k, v in list(stats['top_values'].items())[:3]])
                logging.info(f"    高频值: {top_str}")

class EmbeddingLayer(nn.Module):
    """嵌入层，处理离散特征"""
    
    def __init__(self, 
                 vocab_sizes: Dict[str, int], 
                 embedding_dim: int,
                 sparse_features: List[str]):
        super().__init__()
        self.sparse_features = sparse_features
        self.embedding_dim = embedding_dim
        
        # 创建嵌入层
        self.embeddings = nn.ModuleDict()
        for i, feature in enumerate(sparse_features):
            if feature in vocab_sizes:
                vocab_size = vocab_sizes[feature]
                self.embeddings[feature] = nn.Embedding(vocab_size, embedding_dim)
                # 初始化嵌入权重
                nn.init.xavier_uniform_(self.embeddings[feature].weight)
        
        self.total_embedding_dim = len(self.embeddings) * embedding_dim
        
    def forward(self, sparse_inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sparse_inputs: [batch_size, num_sparse_features]
        Returns:
            embedded_features: [batch_size, total_embedding_dim]
        """
        embeddings = []
        
        for i, feature in enumerate(self.sparse_features):
            if feature in self.embeddings:
                # 获取特征值
                feature_values = sparse_inputs[:, i]
                # 嵌入
                embedded = self.embeddings[feature](feature_values)
                embeddings.append(embedded)
        
        if embeddings:
            # 拼接所有嵌入
            return torch.cat(embeddings, dim=1)
        else:
            # 如果没有嵌入，返回零张量
            batch_size = sparse_inputs.size(0)
            return torch.zeros(batch_size, 0, device=sparse_inputs.device)

def create_dsp_feature_processor() -> FeatureProcessor:
    """创建DSP特征处理器"""
    
    # DSP广告特征定义
    dense_features = [
        'ssp_cnt','supply_iap_price_min','supply_iap_price_max','supply_released_days',
        'supply_last_update_days','supply_content_rating','supply_normal_rate',
        'supply_rating_count','supply_real_installs','supply_file_size_bytes',
        'bid_floor','demand_iap_price_min','demand_iap_price_max','demand_released_days',
        'demand_last_update_days','demand_content_rating','demand_normal_rate',
        'demand_rating_count','demand_real_installs','demand_file_size_bytes'
    ]
    
    sparse_features = [
        'hour','weekday','adv_id','affiliate_id','campaign_id','ad_group_id','ad_id',
        'creative_id','feature_1','pos','instl','response_type','ad_format','os',
        'device_make','bundle_id','country','package','category','connection_type',
        'device_model','lang','publisher_id','first_ssp','last_ssp','video_placement',
        'is_rewarded','offer_id','supply_developer_id','supply_genreId','supply_version',
        'supply_minimum_os_version','supply_industry_id','is_oem','tag_id','osv','ua',
        'demand_developer_id','demand_genreId','demand_version','demand_minimum_os_version',
        'demand_industry_id','ip','device_id','ad_width','ad_height'
    ]
    
    return FeatureProcessor(
        dense_features=dense_features,
        sparse_features=sparse_features,
        embedding_dim=8,  # 嵌入维度
        normalize_dense=True,
        handle_unknown='ignore'
    )

def create_sample_dsp_data(num_samples: int = 10000, num_scenes: int = 5) -> pd.DataFrame:
    """创建DSP示例数据"""
    np.random.seed(42)
    
    # DSP特征定义
    dense_features = [
        'ssp_cnt','supply_iap_price_min','supply_iap_price_max','supply_released_days',
        'supply_last_update_days','supply_content_rating','supply_normal_rate',
        'supply_rating_count','supply_real_installs','supply_file_size_bytes',
        'bid_floor','demand_iap_price_min','demand_iap_price_max','demand_released_days',
        'demand_last_update_days','demand_content_rating','demand_normal_rate',
        'demand_rating_count','demand_real_installs','demand_file_size_bytes'
    ]
    
    sparse_features = [
        'hour','weekday','adv_id','affiliate_id','campaign_id','ad_group_id','ad_id',
        'creative_id','feature_1','pos','instl','response_type','ad_format','os',
        'device_make','bundle_id','country','package','category','connection_type',
        'device_model','lang','publisher_id','first_ssp','last_ssp','video_placement',
        'is_rewarded','offer_id','supply_developer_id','supply_genreId','supply_version',
        'supply_minimum_os_version','supply_industry_id','is_oem','tag_id','osv','ua',
        'demand_developer_id','demand_genreId','demand_version','demand_minimum_os_version',
        'demand_industry_id','ip','device_id','ad_width','ad_height'
    ]
    
    data = {}
    
    # 生成连续特征
    for feature in dense_features:
        if 'price' in feature:
            # 价格特征，对数正态分布
            data[feature] = np.random.lognormal(0, 1, num_samples)
        elif 'days' in feature:
            # 天数特征，指数分布
            data[feature] = np.random.exponential(30, num_samples)
        elif 'rating' in feature or 'rate' in feature:
            # 评分特征，0-5之间
            data[feature] = np.random.uniform(0, 5, num_samples)
        elif 'count' in feature or 'installs' in feature:
            # 计数特征，幂律分布
            data[feature] = np.random.pareto(1, num_samples) * 1000
        elif 'size' in feature:
            # 文件大小，对数正态分布
            data[feature] = np.random.lognormal(15, 2, num_samples)  # 字节
        elif feature == 'ssp_cnt':
            # SSP数量
            data[feature] = np.random.poisson(3, num_samples)
        elif feature == 'bid_floor':
            # 底价
            data[feature] = np.random.gamma(2, 0.5, num_samples)
        else:
            # 其他连续特征
            data[feature] = np.random.normal(0, 1, num_samples)
    
    # 生成离散特征
    for feature in sparse_features:
        if feature == 'hour':
            data[feature] = np.random.randint(0, 24, num_samples)
        elif feature == 'weekday':
            data[feature] = np.random.randint(0, 7, num_samples)
        elif feature in ['pos', 'instl', 'is_rewarded', 'is_oem']:
            # 二值特征
            data[feature] = np.random.randint(0, 2, num_samples)
        elif feature in ['ad_width', 'ad_height']:
            # 广告尺寸
            if feature == 'ad_width':
                data[feature] = np.random.choice([320, 728, 300, 250], num_samples)
            else:
                data[feature] = np.random.choice([50, 90, 250, 250], num_samples)
        elif 'id' in feature or feature in ['bundle_id', 'package', 'device_id', 'ip']:
            # ID类特征，高基数
            if feature in ['device_id', 'ip']:
                cardinality = min(num_samples // 2, 50000)  # 设备ID和IP基数很高
            elif feature in ['adv_id', 'campaign_id', 'ad_group_id', 'ad_id', 'creative_id']:
                cardinality = min(num_samples // 10, 5000)  # 广告相关ID
            else:
                cardinality = min(num_samples // 5, 1000)  # 其他ID
            data[feature] = np.random.randint(0, cardinality, num_samples)
        elif feature == 'country':
            # 国家代码
            countries = ['US', 'CN', 'JP', 'DE', 'UK', 'FR', 'KR', 'IN', 'BR', 'RU']
            data[feature] = np.random.choice(countries, num_samples)
        elif feature == 'os':
            # 操作系统
            os_list = ['Android', 'iOS']
            data[feature] = np.random.choice(os_list, num_samples)
        elif feature == 'device_make':
            # 设备制造商
            makes = ['Samsung', 'Apple', 'Huawei', 'Xiaomi', 'Oppo', 'Vivo', 'OnePlus']
            data[feature] = np.random.choice(makes, num_samples)
        elif feature == 'connection_type':
            # 连接类型
            conn_types = ['WiFi', '4G', '5G', '3G']
            data[feature] = np.random.choice(conn_types, num_samples)
        elif feature == 'ad_format':
            # 广告格式
            formats = ['banner', 'interstitial', 'rewarded_video', 'native']
            data[feature] = np.random.choice(formats, num_samples)
        elif feature == 'response_type':
            # 响应类型
            response_types = ['cpm', 'cpc', 'cpa']
            data[feature] = np.random.choice(response_types, num_samples)
        elif feature == 'video_placement':
            # 视频位置
            placements = ['pre_roll', 'mid_roll', 'post_roll', 'standalone']
            data[feature] = np.random.choice(placements, num_samples)
        elif feature in ['category', 'supply_genreId', 'demand_genreId']:
            # 类别特征
            data[feature] = np.random.randint(0, 50, num_samples)
        elif feature == 'lang':
            # 语言
            languages = ['en', 'zh', 'ja', 'ko', 'de', 'fr', 'es', 'pt', 'ru', 'ar']
            data[feature] = np.random.choice(languages, num_samples)
        else:
            # 其他离散特征，中等基数
            data[feature] = np.random.randint(0, 100, num_samples)
    
    # 生成场景ID
    data['scene_id'] = np.random.randint(0, num_scenes, num_samples)
    
    # 生成目标变量（基于特征的复杂组合）
    # CTR模型
    ctr_signal = (
        0.1 * np.log1p(data['bid_floor']) +
        0.05 * data['supply_normal_rate'] +
        0.03 * (data['hour'] < 12).astype(float) +
        0.02 * (data['weekday'] < 5).astype(float) +
        np.random.normal(0, 0.5, num_samples)
    )
    ctr_probs = 1 / (1 + np.exp(-ctr_signal))
    data['ctr'] = (np.random.random(num_samples) < ctr_probs).astype(float)
    
    # CVR模型（只在点击的情况下）
    cvr_signal = (
        0.08 * data['demand_normal_rate'] +
        0.05 * np.log1p(data['demand_real_installs']) +
        0.03 * (data['ad_format'] == 'rewarded_video').astype(float) +
        np.random.normal(0, 0.3, num_samples)
    )
    cvr_probs = 1 / (1 + np.exp(-cvr_signal))
    cvr_raw = (np.random.random(num_samples) < cvr_probs).astype(float)
    # CVR只在有点击的情况下有意义
    data['cvr'] = np.where(data['ctr'] == 1, cvr_raw, np.nan)
    
    # IVR模型（曝光到转化）
    ivr_signal = (
        0.06 * data['supply_normal_rate'] +
        0.04 * data['demand_normal_rate'] +
        0.02 * np.log1p(data['bid_floor']) +
        np.random.normal(0, 0.4, num_samples)
    )
    ivr_probs = 1 / (1 + np.exp(-ivr_signal)) * 0.05  # IVR通常很低
    data['ivr'] = (np.random.random(num_samples) < ivr_probs).astype(float)
    
    return pd.DataFrame(data)

# 特征重要性分析
def analyze_feature_importance(df: pd.DataFrame, 
                             target_col: str,
                             dense_features: List[str],
                             sparse_features: List[str]) -> Dict:
    """分析特征重要性"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    
    # 准备数据
    feature_importance = {}
    
    # 处理连续特征
    X_dense = df[dense_features].fillna(0)
    
    # 处理离散特征
    X_sparse = df[sparse_features].copy()
    encoders = {}
    for col in sparse_features:
        if col in X_sparse.columns:
            le = LabelEncoder()
            X_sparse[col] = le.fit_transform(X_sparse[col].fillna('unknown').astype(str))
            encoders[col] = le
    
    # 合并特征
    X = pd.concat([X_dense, X_sparse], axis=1)
    y = df[target_col].fillna(0)
    
    # 训练随机森林
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # 获取特征重要性
    feature_names = list(X.columns)
    importances = rf.feature_importances_
    
    feature_importance = dict(zip(feature_names, importances))
    
    # 按重要性排序
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'feature_importance': dict(sorted_features),
        'top_10_features': [f[0] for f in sorted_features[:10]]
    }