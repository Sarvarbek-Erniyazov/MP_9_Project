import json
import requests
import pandas as pd
import gradio as gr

import os
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict_from_scaled")



def call_api(features_json: str):
    """
    User JSON -> FastAPI -> prediction
    JSON format:
    {
      "FE_lag_24h": 0.0,
      ...
    }
    """
    try:
        features = json.loads(features_json)
        if not isinstance(features, dict):
            return "❌ JSON dict bo‘lishi kerak: {\"FE_lag_24h\": 0.1, ...}"
    except Exception as e:
        return f"❌ JSON parse xato: {e}"

    payload = {"features": features}

    try:
        r = requests.post(API_URL, json=payload, timeout=20)
        if r.status_code != 200:
            return f"❌ API error {r.status_code}: {r.text}"
        data = r.json()
        return f"✅ Prediction: {data['prediction']}"
    except Exception as e:
        return f"❌ API ga ulanish xato: {e}"


def from_csv(file, row_index: int):
    """
    Upload test_scaled.csv (yoki shunga o‘xshash)
    row_index qatordan feature olib API ga yuboradi.
    """
    try:
        df = pd.read_csv(file.name)
        if "Datetime" in df.columns:
            df = df.drop(columns=["Datetime"])
        if "PJME_MW" in df.columns:
            df = df.drop(columns=["PJME_MW"])

        if row_index < 0 or row_index >= len(df):
            return f"❌ row_index 0..{len(df)-1} oralig‘ida bo‘lsin"

        features = df.iloc[row_index].to_dict()
        payload = {"features": {k: float(v) for k, v in features.items()}}

        r = requests.post(API_URL, json=payload, timeout=20)
        if r.status_code != 200:
            return f"❌ API error {r.status_code}: {r.text}"
        data = r.json()
        return f"✅ Prediction: {data['prediction']}"
    except Exception as e:
        return f"❌ CSV dan o‘qish/yuborish xato: {e}"


with gr.Blocks(title="Energy Forecasting Demo") as demo:
    gr.Markdown("## Energy Forecasting (Gradio) — FastAPI orqali predict")

    with gr.Tab("JSON bilan (scaled features)"):
        gr.Markdown("11 ta feature JSON ko‘rinishida yuboring (test_scaled.csv dagi ustunlar).")
        json_in = gr.Textbox(lines=14, label="features JSON", value='{\n  "FE_lag_24h": 0.0,\n  "FE_lag_168h": 0.0,\n  "FE_rolling_mean_24h": 0.0,\n  "Temp_K": 0.0,\n  "FE_Temp_K_sq": 0.0,\n  "FE_dayofyear": 0.0,\n  "FE_hour": 0.0,\n  "FE_month": 0.0,\n  "FE_quarter": 0.0,\n  "FE_dayofweek": 0.0,\n  "FE_year": 0.0\n}')
        out1 = gr.Textbox(label="Natija")
        btn1 = gr.Button("Predict")
        btn1.click(call_api, inputs=json_in, outputs=out1)

    with gr.Tab("CSV upload (test_scaled.csv)"):
        gr.Markdown("test_scaled.csv ni upload qiling, qaysi qatordan olishni tanlang.")
        file_in = gr.File(label="CSV file")
        row_in = gr.Number(value=0, precision=0, label="row_index (0 dan boshlanadi)")
        out2 = gr.Textbox(label="Natija")
        btn2 = gr.Button("Predict from CSV row")
        btn2.click(from_csv, inputs=[file_in, row_in], outputs=out2)

demo.launch(server_name="0.0.0.0", server_port=7860)
