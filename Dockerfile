FROM python:3.9-slim

WORKDIR /app

# Tizim paketlari
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Kutubxonalarni o'rnatish
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Barcha papkalarni (api va ui) konteynerga nusxalash
COPY . .

# Portlar
EXPOSE 8000
EXPOSE 7860

# CMD-ni sizning papkalaringizga mosladik
CMD ["sh", "-c", "python api/main.py & python ui/app.py"]