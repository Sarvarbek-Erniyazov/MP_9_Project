import os
import json
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from src.logger import logging_instance
from sklearn.ensemble import RandomForestRegressor

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

def objective(trial, X, y, model_name):
    if model_name == "LightGBM":
        params = {'n_estimators': trial.suggest_int('n_estimators', 500, 1000),
                  'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                  'num_leaves': trial.suggest_int('num_leaves', 20, 100)}
        model = LGBMRegressor(**params, verbosity=-1)
    elif model_name == "XGBoost":
        params = {'n_estimators': trial.suggest_int('n_estimators', 500, 1000),
                  'max_depth': trial.suggest_int('max_depth', 3, 9),
                  'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True)}
        model = XGBRegressor(**params)
    else:
        params = {'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                  'max_depth': trial.suggest_int('max_depth', 5, 15)}
        model = RandomForestRegressor(**params)

    score = cross_val_score(model, X, y, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
    return -score

def run_all_tuning(train_path, config_path="configs/best_params.json"):
    df = pd.read_csv(train_path)
    X = df.drop(columns=['PJME_MW', 'Datetime'], errors='ignore')
    y = df['PJME_MW']
    
    os.makedirs("configs", exist_ok=True)
    all_best_params = {}
    
    tune_list = ["RandomForest"]
    if HAS_XGB: tune_list.append("XGBoost")
    if HAS_LGBM: tune_list.append("LightGBM")

    for name in tune_list:
        logging_instance.info(f"--- {name} tuning boshlandi ---")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X, y, name), n_trials=15)
        all_best_params[name] = study.best_params

    with open(config_path, 'w') as f:
        json.dump(all_best_params, f, indent=4)