from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Energy Forecasting API")

# Modellarni yuklash
MODEL_PATH = "models/final_stacking_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/selected_features.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model fayllari topilmadi! Avval run.py ni ishga tushiring.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)

# Kiruvchi ma'lumot strukturasi
class PredictionInput(BaseModel):
    # Modelga kerakli asosiy xususiyatlar (Input sifatida keladi)
    Temp_K: float
    FE_lag_24h: float
    FE_lag_168h: float
    FE_rolling_mean_24h: float
    FE_hour: int
    FE_dayofweek: int
    FE_month: int
    FE_quarter: int
    FE_year: int
    FE_dayofyear: int
    FE_Temp_K_sq: float

@app.get("/")
def home():
    return {"message": "Energy Forecasting API ishlamoqda!"}

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        # 1. Kelgan ma'lumotni DataFrame-ga o'tkazish
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])
        
        # 2. Ustunlar tartibini model o'rgangan tartibga keltirish
        input_df = input_df[features]
        
        # 3. Scaling
        input_scaled = scaler.transform(input_df)
        
        # 4. Prediction
        prediction = model.predict(input_scaled)
        
        return {
            "prediction_mw": float(prediction[0]),
            "unit": "MW"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))