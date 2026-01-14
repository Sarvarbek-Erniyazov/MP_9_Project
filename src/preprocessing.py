import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from src.logger import logging_instance

def run_preprocessing(raw_path, output_dir="data/processed"):
    logging_instance.info("0-QADAM: Preprocessing va Feature Engineering boshlandi.")
    df = pd.read_csv(raw_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime')

    # Feature Engineering
    df['FE_hour'] = df['Datetime'].dt.hour
    df['FE_dayofweek'] = df['Datetime'].dt.dayofweek
    df['FE_month'] = df['Datetime'].dt.month
    df['FE_dayofyear'] = df['Datetime'].dt.dayofyear
    
    # Lag features
    df['FE_lag_24h'] = df['PJME_MW'].shift(24)
    df['FE_lag_168h'] = df['PJME_MW'].shift(168)
    df['FE_rolling_mean_24h'] = df['PJME_MW'].shift(1).rolling(window=24).mean()
    
    # Sun'iy ob-havo xususiyati (leverage uchun)
    df['FE_Temp_K_sq'] = np.power(df['FE_month'] - 7, 2) 

    df = df.dropna().reset_index(drop=True)

    # Time-based splitting (Data Leakage oldini olish)
    split_date = df['Datetime'].max() - pd.Timedelta(days=365)
    train = df[df['Datetime'] <= split_date].copy()
    test = df[df['Datetime'] > split_date].copy()

    # Scaling
    num_cols = ['FE_lag_24h', 'FE_lag_168h', 'FE_rolling_mean_24h', 'FE_Temp_K_sq']
    scaler = StandardScaler()
    train[num_cols] = scaler.fit_transform(train[num_cols])
    test[num_cols] = scaler.transform(test[num_cols])

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_final.csv")
    test_path = os.path.join(output_dir, "test_final.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    
    return train_path, test_path