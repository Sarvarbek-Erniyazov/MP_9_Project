FROM python:3.12-slim

# LightGBM va XGBoost uchun system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency cache uchun avval requirements
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Project kodlari
COPY app/ app/
COPY demo/ demo/
COPY models/ models/
COPY data/processed/ data/processed/

ENV MODEL_PATH=models/final_stacking_model.pkl
ENV API_URL=http://127.0.0.1:8000/predict_from_scaled
EXPOSE 8000
EXPOSE 7860

CMD ["bash", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & python demo/gradio_app.py"]
