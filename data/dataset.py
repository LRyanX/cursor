from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import pytorch_lightning as pl


ALL_FEATURES: List[str] = [
    'hour','weekday','adv_id','affiliate_id','campaign_id','ad_group_id','ad_id',
    'creative_id','feature_1','pos','instl','response_type','ad_format','os',
    'device_make','bundle_id','country','package','category','connection_type',
    'device_model','lang','publisher_id','first_ssp','last_ssp','video_placement',
    'is_rewarded','offer_id','supply_developer_id','supply_genreId','supply_version',
    'supply_minimum_os_version','supply_industry_id','is_oem','tag_id','osv','ua',
    'demand_developer_id','demand_genreId','demand_version','demand_minimum_os_version',
    'demand_industry_id','ip','device_id','ad_width','ad_height'
]
LABEL_COLS = ['ctr_label', 'cvr_label', 'ivr_label']


class CSVDataset(Dataset):
    def __init__(self, df: pd.DataFrame, encoders: Dict[str, LabelEncoder]):
        self.labels = df[LABEL_COLS].astype(np.float32).values
        fields = []
        for col in ALL_FEATURES:
            if col not in encoders:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
            else:
                df[col] = encoders[col].transform(df[col].astype(str))
            fields.append(df[col].values)
        self.features = np.stack(fields, axis=1).astype(np.int64)
        self.encoders = encoders

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx])
        y = torch.from_numpy(self.labels[idx])
        return x, y


class DSPDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 1024, num_workers: int = 4, val_ratio: float = 0.1):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.encoders: Dict[str, LabelEncoder] = {}

    def setup(self, stage: str = None):
        df = pd.read_parquet(self.data_path) if self.data_path.endswith(".parquet") else pd.read_csv(self.data_path)
        # shuffle before split
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        val_len = int(len(df) * self.val_ratio)
        df_val = df.iloc[:val_len]
        df_train = df.iloc[val_len:]
        self.train_ds = CSVDataset(df_train, self.encoders)
        self.val_ds = CSVDataset(df_val, self.encoders)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)