import json
import os
import pandas as pd
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from src.logger import logging_instance

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

def create_stacking_ensemble(params_path="configs/best_params.json"):
    if not os.path.exists(params_path):
        logging_instance.error(f"Parametrlar fayli topilmadi: {params_path}")
        raise FileNotFoundError(f"{params_path} mavjud emas!")

    with open(params_path, 'r') as f:
        best_params = json.load(f)

    logging_instance.info("Stacking Ensemble uchun bazaviy modellar yig'ilmoqda...")

    # Bazaviy o'rganuvchilar
    base_learners = []
    
    if HAS_LGBM:
        base_learners.append(('lgbm', LGBMRegressor(**best_params['LightGBM'], verbosity=-1)))
    
    if HAS_XGB:
        base_learners.append(('xgb', XGBRegressor(**best_params['XGBoost'])))
        
    base_learners.append(('rf', RandomForestRegressor(**best_params['RandomForest'])))

    # Stacking modeli
    stack_model = StackingRegressor(
        estimators=base_learners,
        final_estimator=RidgeCV(), # Meta-learner (Linear Regression L2 bilan)
        cv=5,
        n_jobs=-1
    )

    return stack_model