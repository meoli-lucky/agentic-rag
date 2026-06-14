FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Cài đặt các thư viện đồ họa và nền hệ thống bắt buộc cho OpenCV/Paddle
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Cài đặt các thư viện Python (Phiên bản 2.8.1 bọc tốt cả v4 lẫn v5 core)
RUN pip3 install --no-cache-dir \
    fastapi>=0.110.0 \
    uvicorn[standard]>=0.22.0 \
    python-multipart>=0.0.9 \
    paddlepaddle-gpu==2.6.1 \
    paddleocr==2.8.1

# 3. Copy file code vào container (Mặc dù khi chạy Production ta vẫn gá đè bằng cờ -v)
COPY server.py .

# 4. THIẾT LẬP CÁC BIẾN MÔI TRƯỜNG ĐIỀU PHỐI MẶC ĐỊNH
ENV OCR_NUM_WORKERS=1
ENV OCR_VERSION=PP-OCRv4
ENV OCR_USE_ANGLE_CLS=false
ENV OCR_LANG=vi
ENV OCR_GPU_ID=1
ENV OCR_ALLOWED_EXTENSIONS=.png,.jpg,.jpeg,.tiff,.bmp,.webp

# 5. Mở cổng mạng nội bộ của ứng dụng theo đặc tả cấu hình
EXPOSE 8201

# 6. Khởi chạy Uvicorn, tự động bốc số lượng workers từ môi trường ngoài vào
CMD uvicorn server:app --host 0.0.0.0 --port 8201 --workers $OCR_NUM_WORKERS