import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

# =============================
# PATHLAR (SENING KOMPYUTERING)
# =============================
TEST_SCALED_PATH = r"C:\Users\sarva\Desktop\energy_forecasting_pro\data\processed\test_scaled.csv"
MODEL_PATH = r"C:\Users\sarva\Desktop\energy_forecasting_pro\models\final_stacking_model.pkl"

TARGET_COL = "PJME_MW"
DATETIME_COL = "Datetime"


def metrics(y_true, y_pred):
    return {
        "MAPE(%)": round(mean_absolute_percentage_error(y_true, y_pred) * 100, 3),
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
    }


def main():
    # 1) test_scaled.csv ni o‘qiymiz
    df = pd.read_csv(TEST_SCALED_PATH)

    if DATETIME_COL not in df.columns:
        raise ValueError("❌ test_scaled.csv ichida 'Datetime' ustuni yo‘q")

    if TARGET_COL not in df.columns:
        raise ValueError("❌ test_scaled.csv ichida 'PJME_MW' ustuni yo‘q")

    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    df = df.sort_values(DATETIME_COL)

    # 2) Modelni yuklaymiz
    model = joblib.load(MODEL_PATH)

    # 3) Feature'larni ajratamiz
    feature_cols = [c for c in df.columns if c not in [DATETIME_COL, TARGET_COL]]
    X = df[feature_cols]
    y = df[TARGET_COL].values

    # 4) Predict
    preds = model.predict(X)

    # 5) Natijalar
    report = []

    # OVERALL
    report.append({"Slice": "OVERALL", **metrics(y, preds)})

    # WEEKDAY / WEEKEND
    is_weekend = df[DATETIME_COL].dt.dayofweek >= 5
    report.append({
        "Slice": "WEEKDAY",
        **metrics(y[~is_weekend], preds[~is_weekend])
    })
    report.append({
        "Slice": "WEEKEND",
        **metrics(y[is_weekend], preds[is_weekend])
    })

    # PEAK HOURS (17–21)
    hours = df[DATETIME_COL].dt.hour
    peak_mask = (hours >= 17) & (hours <= 21)
    report.append({
        "Slice": "PEAK_17_21",
        **metrics(y[peak_mask], preds[peak_mask])
    })

    # 6) Print
    result_df = pd.DataFrame(report)
    print("\n================ STEP-0 EVALUATION REPORT ================\n")
    print(result_df.to_string(index=False))
    print(f"\nUsed features count: {len(feature_cols)}")
    print("\n==========================================================\n")


if __name__ == "__main__":
    main()
