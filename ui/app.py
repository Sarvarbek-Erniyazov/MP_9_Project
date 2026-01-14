import gradio as gr
import requests
import datetime

def predict_energy(temp, lag_24h, lag_168h, rolling_24h, date_input):
    # Sanadan xususiyatlarni ajratib olish
    dt = datetime.datetime.strptime(date_input, "%Y-%m-%d %H:%M")
    
    # API-ga yuboriladigan ma'lumot
    payload = {
        "Temp_K": temp,
        "FE_lag_24h": lag_24h,
        "FE_lag_168h": lag_168h,
        "FE_rolling_mean_24h": rolling_24h,
        "FE_hour": dt.hour,
        "FE_dayofweek": dt.weekday(),
        "FE_month": dt.month,
        "FE_quarter": (dt.month - 1) // 3 + 1,
        "FE_year": dt.year,
        "FE_dayofyear": dt.timetuple().tm_yday,
        "FE_Temp_K_sq": temp ** 2
    }
    
    try:
        # FastAPI serveriga POST so'rov yuborish
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        prediction = response.json()["prediction_mw"]
        return f"Bashorat qilingan energiya iste'moli: {prediction:.2f} MW"
    except Exception as e:
        return f"Xatolik yuz berdi: {str(e)}"

# Gradio interfeysi
with gr.Blocks(title="Energy Forecast System") as demo:
    gr.Markdown("# ⚡ Energy Forecasting System (Stacking Ensemble)")
    
    with gr.Row():
        with gr.Column():
            temp = gr.Slider(250, 320, value=290, label="Harorat (Kelvin)")
            lag_24h = gr.Number(label="24 soat oldingi iste'mol (MW)", value=30000)
            lag_168h = gr.Number(label="1 hafta oldingi iste'mol (MW)", value=30000)
            rolling_24h = gr.Number(label="O'rtacha kunlik iste'mol (MW)", value=30000)
            date_input = gr.Textbox(label="Sana va Vaqt (YYYY-MM-DD HH:mm)", value="2026-01-14 12:00")
            
        with gr.Column():
            output = gr.Textbox(label="Natija")
            btn = gr.Button("Bashorat qilish", variant="primary")
            
    btn.click(fn=predict_energy, inputs=[temp, lag_24h, lag_168h, rolling_24h, date_input], outputs=output)

if __name__ == "__main__":
    demo.launch()