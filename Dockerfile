FROM python:3.9-slim

WORKDIR /app

# Tizim kutubxonalari (LightGBM uchun kerak)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Fayl nomi requirements.txt ekanligiga e'tibor bering!
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE 7860

CMD ["sh", "-c", "python api/main.py & python app.py"]