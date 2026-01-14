FROM python:3.9-slim

WORKDIR /app

# Tizim kutubxonalari (LightGBM yoki XGBoost uchun kerak bo'lishi mumkin)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Hamma kodni nusxalash
COPY . .

EXPOSE 8000
EXPOSE 7860

# FastAPI va Gradio-ni birga yurgizish
CMD ["sh", "-c", "python api/main.py & python app.py"]