import os
import joblib
import pandas as pd
import numpy as np
from src.logger import logging_instance
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Smart Imports
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

def train_base_models(train_path, test_path):
    try:
        logging_instance.info("--- BASE MODEL TRAINING BOSHLANDI ---")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        X_train = train_df.drop(columns=['PJME_MW', 'Datetime'], errors='ignore')
        y_train = train_df['PJME_MW']
        X_test = test_df.drop(columns=['PJME_MW', 'Datetime'], errors='ignore')
        y_test = test_df['PJME_MW']

        models = {}
        if HAS_XGB:
            models["XGBoost"] = XGBRegressor(n_estimators=500, learning_rate=0.1, random_state=42)
        if HAS_LGBM:
            models["LightGBM"] = LGBMRegressor(n_estimators=500, learning_rate=0.1, random_state=42, verbose=-1)
        
        models["RandomForest"] = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

        results = {}
        for name, model in models.items():
            logging_instance.info(f"{name} o'qitilmoqda...")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, preds) * 100
            results[name] = mape
            print(f"📊 {name} Baseline MAPE: {mape:.2f}%")
        
        return results
    except Exception as e:
        logging_instance.error(f"Trainer xatosi: {e}")
        raise e