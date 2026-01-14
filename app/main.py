from __future__ import annotations

import os
from typing import Dict, List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "models/final_stacking_model.pkl")


FEATURES: List[str] = [
    "FE_lag_24h",
    "FE_lag_168h",
    "FE_rolling_mean_24h",
    "Temp_K",
    "FE_Temp_K_sq",
    "FE_dayofyear",
    "FE_hour",
    "FE_month",
    "FE_quarter",
    "FE_dayofweek",
    "FE_year",
]


class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="11 ta feature qiymatlari. Masalan: {'FE_lag_24h': 0.12, ...}"
    )


class PredictResponse(BaseModel):
    prediction: float
    used_features: List[str]


app = FastAPI(title="Energy Forecasting API (MVP)", version="0.1.0")

MODEL = None


@app.on_event("startup")
def load_model():
    global MODEL
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model topilmadi: {MODEL_PATH}")
    MODEL = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict_from_scaled", response_model=PredictResponse)
def predict_from_scaled(req: PredictRequest):
    global MODEL
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model yuklanmagan")

    missing = [f for f in FEATURES if f not in req.features]
    extra = [k for k in req.features.keys() if k not in FEATURES]

    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")
    if extra:
        raise HTTPException(status_code=422, detail=f"Unknown features: {extra}")

    x = np.array([[req.features[f] for f in FEATURES]], dtype=float)

    try:
        pred = float(MODEL.predict(x)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    return PredictResponse(prediction=pred, used_features=FEATURES)
