# Usar una versión ligera de Python
FROM python:3.10-slim

# Directorio de trabajo en el servidor
WORKDIR /app

# Instalar Tesseract OCR, español y herramientas de descarga
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-spa \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copiar requerimientos
COPY requirements.txt .

# TRUCO VITAL: Instalar PyTorch en su versión CPU primero para no colapsar la RAM
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Instalar el resto de librerías
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código de tu proyecto
COPY . .

# (OPCIONAL) Descargar el modelo pesado si no lo pudiste subir a GitHub.
# Descomenta la siguiente línea y pon tu link de descarga directa:
# RUN wget -O models/fme-layoutlmv3-funsd.zip "TU_LINK_DE_DESCARGA_AQUI"

# Ejecutar FastAPI en el puerto dinámico que Railway asigne
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT