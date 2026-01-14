import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error

# Modullarni import qilish
from src.logger import logging_instance
from src.ingestion import DataIngestor
from src.feature_engineering import FeatureEngineer
from src.feature_selection import FeatureSelector
from src.model_trainer import train_base_models
from src.tuner import run_all_tuning
from src.ensemble import create_stacking_ensemble

def main():
    try:
        # 1. KONFIGURATSIYA
        RAW_ENERGY = "data/raw/PJME_hourly.csv"
        RAW_TEMP = "data/raw/temperature.csv"
        COMBINED_DIR = "data/combined"
        PROCESSED_DIR = "data/processed"
        MODEL_DIR = "models"
        CONFIG_PATH = "configs/best_params.json"
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        
        logging_instance.info("--- INTEGRATSIYALASHGAN PIPELINE BOSHLANDI ---")

        # 2. DATA INGESTION
        ingestor = DataIngestor(RAW_ENERGY, RAW_TEMP, COMBINED_DIR)
        ingestor.run_ingestion()
        combined_path = os.path.join(COMBINED_DIR, "combined_data.csv")
        combined_df = pd.read_csv(combined_path)

        # 3. FEATURE ENGINEERING
        engineer = FeatureEngineer(split_date='2017-01-01')
        train_df, test_df = engineer.run_feature_engineering(combined_df)

        # 4. FEATURE SELECTION
        X_selector = train_df.drop(columns=['PJME_MW'])
        y_selector = train_df['PJME_MW']
        
        selector = FeatureSelector()
        selected_features, _ = selector.analyze_importance(X_selector, y_selector)
        
        # Tanlangan feature'larni saqlash (Keyinchalik prediction uchun kerak)
        joblib.dump(selected_features, os.path.join(MODEL_DIR, "selected_features.pkl"))
        logging_instance.info(f"Tanlangan xususiyatlar saqlandi: {len(selected_features)} ta")

        # 5. DATA PREPARATION & SCALING
        X_train = train_df[selected_features]
        y_train = train_df['PJME_MW']
        X_test = test_df[selected_features]
        y_test = test_df['PJME_MW']

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=selected_features, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=selected_features, index=X_test.index)
        
        # Scalerni saqlash
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

        # Fayllarni saqlash (Base trainer o'qishi uchun)
        train_path_scaled = os.path.join(PROCESSED_DIR, "train_scaled.csv")
        test_path_scaled = os.path.join(PROCESSED_DIR, "test_scaled.csv")
        pd.concat([X_train_scaled, y_train], axis=1).to_csv(train_path_scaled)
        pd.concat([X_test_scaled, y_test], axis=1).to_csv(test_path_scaled)

        # 6. BASE MODELS & TUNING
        base_results = train_base_models(train_path_scaled, test_path_scaled)
        run_all_tuning(train_path_scaled, CONFIG_PATH)

        # 7. STACKING ENSEMBLE
        stack_model = create_stacking_ensemble(CONFIG_PATH)
        logging_instance.info("Kuchaytirilgan Stacking Ensemble o'qitilmoqda...")
        stack_model.fit(X_train_scaled, y_train)
        
        # Yakuniy modelni saqlash
        joblib.dump(stack_model, os.path.join(MODEL_DIR, "final_stacking_model.pkl"))
        logging_instance.info(f"Model '{MODEL_DIR}/' papkasiga saqlandi.")

        # 8. NATIJALARNI HISOBLASH
        stack_preds = stack_model.predict(X_test_scaled)
        stack_mape = mean_absolute_percentage_error(y_test, stack_preds) * 100

        # NATIJALARNI SOLISHTIRISH
        comparison = []
        for model, mape in base_results.items():
            comparison.append({"Model": model, "Holat": "Baseline", "MAPE (%)": f"{mape:.2f}%"})
        comparison.append({"Model": "Stacking", "Holat": "Optimized", "MAPE (%)": f"{stack_mape:.2f}%"})
        
        print("\n" + "="*65)
        print("📊 PIPELINE YAKUNLANDI. NATIJALAR:")
        print(pd.DataFrame(comparison).to_string(index=False))
        print(f"\n📂 Barcha modellar '{MODEL_DIR}/' papkasida saqlandi.")
        print("="*65)

    except Exception as e:
        logging_instance.error(f"Pipeline xatosi: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()