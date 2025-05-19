FROM python:3.11-slim

# Install system dependencies for OpenCV and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \  # Critical for OpenCV
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Pin OpenCV version explicitly
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    opencv-python-headless==4.8.1.78 \
    numpy==1.26.4 \
    streamlit==1.24.1 \
    Pillow==10.0.0 \
    ultralytics==8.0.0

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
