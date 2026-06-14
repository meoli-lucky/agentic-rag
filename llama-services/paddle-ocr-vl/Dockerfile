# Sử dụng CUDA base image để hỗ trợ tăng tốc GPU phần cứng
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Thiết lập biến môi trường
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

# Cài đặt Python và các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    cmake \
    git \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Tạo symlink cho python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Nâng cấp pip
RUN pip install --no-cache-dir --upgrade pip

# Biên dịch llama-cpp-python với CUDA backend
# Cờ này quan trọng để offload các tensor lên VRAM, tối ưu cho xử lý ảnh/text tốc độ cao
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1
RUN pip install --no-cache-dir 'llama-cpp-python[server]' pydantic sse-starlette starlette-context

# Tạo thư mục làm việc
WORKDIR /app
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose port API
EXPOSE 8300

# Khởi chạy script
ENTRYPOINT ["/app/start.sh"]