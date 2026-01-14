import pandas as pd
import numpy as np
import os
from src.logger import logging_instance

class FeatureEngineer:
    def __init__(self, split_date='2017-01-01'):
        self.split_date = split_date

    def run_feature_engineering(self, df):
        try:
            logging_instance.info("2-Bosqich: Feature Engineering boshlandi (Prefikslar bilan).")
            
            df = df.copy()
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df = df.set_index('Datetime').sort_index()

            # 1. Calendar Features (FE_ prefiksi bilan)
            df['FE_hour'] = df.index.hour
            df['FE_dayofweek'] = df.index.dayofweek
            df['FE_month'] = df.index.month
            df['FE_quarter'] = df.index.quarter
            df['FE_year'] = df.index.year
            df['FE_dayofyear'] = df.index.dayofyear
            
            # 2. Weather Square
            df['FE_Temp_K_sq'] = df['Temp_K'] ** 2

            # 3. Lag Features
            df['FE_lag_24h'] = df['PJME_MW'].shift(24)
            df['FE_lag_168h'] = df['PJME_MW'].shift(168)

            # 4. Rolling Window
            df['FE_rolling_mean_24h'] = df['PJME_MW'].shift(1).rolling(window=24).mean()

            # NaN qiymatlarni tozalash
            df = df.dropna()
            
            logging_instance.info(f"Yangi xususiyatlar yaratildi (FE_ prefiksi bilan). Jami ustunlar: {df.shape[1]}")

            # 5. Train va Testga ajratish
            train = df.loc[df.index < self.split_date].copy()
            test = df.loc[df.index >= self.split_date].copy()
            
            logging_instance.info(f"Data Split yakunlandi. Train: {len(train)}, Test: {len(test)}")
            
            return train, test

        except Exception as e:
            logging_instance.error(f"Feature Engineeringda xato: {str(e)}")
            raise e