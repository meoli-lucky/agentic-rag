# Sử dụng base image CUDA + cuDNN của NVIDIA
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Thiết lập biến môi trường để tránh bị hỏi timezone khi cài đặt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Cài đặt Python 3.10 và các thư viện hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Alias python3 thành python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Cập nhật pip
RUN pip install --no-cache-dir --upgrade pip

# Tạo thư mục làm việc
WORKDIR /app

# Copy file requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . /app

# Lệnh chạy mặc định khi khởi động container
CMD ["python", "main.py"]