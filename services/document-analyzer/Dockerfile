# Sử dụng base image tối ưu cho inference
FROM ultralytics/ultralytics:8.4.66-runner

WORKDIR /app

# Cài đặt các package hệ thống cơ bản
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl \
    && rm -rf /var/lib/apt/lists/*

# Copy và cài đặt thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tạo thư mục tải model và tải file .pt của DocLayout-YOLO
# Tạo thư mục và tải TRỰC TIẾP file model .pt của bản DocLayNet
RUN mkdir -p /app/models && \
    wget -O /app/models/doclayout_yolo.pt https://huggingface.co/juliozhao/DocLayout-YOLO-DocLayNet-Docsynth300K_pretrained/resolve/main/doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt

# Copy toàn bộ mã nguồn
COPY . /app

# Tạo thư mục I/O
RUN mkdir -p /app/input /app/output

# Mở port cho FastAPI
EXPOSE 8500

# XÓA BỎ mọi entrypoint ngầm định của base image
ENTRYPOINT []

# Khởi chạy Uvicorn thông qua module Python để đảm bảo tiến trình chạy chuẩn xác
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8500"]