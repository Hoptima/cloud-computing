# Gunakan base image Python
FROM python:3.9-slim

# Set working directory dalam container
WORKDIR /app

# Salin file yang diperlukan ke container
COPY . /app

# Install dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable untuk Flask
ENV FLASK_APP=main.py
ENV FLASK_ENV=production

# Tentukan port yang akan digunakan
EXPOSE 8080

# Jalankan aplikasi Flask
CMD ["python", "main.py"]
