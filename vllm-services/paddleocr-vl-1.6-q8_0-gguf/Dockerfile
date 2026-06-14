# Sử dụng image CUDA chính thức từ NVIDIA làm base
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder

# Cài đặt các công cụ cần thiết (Bổ sung ccache và libcurl cho bản mới)
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    ccache \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone source code llama.cpp và build kèm hỗ trợ CUDA
WORKDIR /app
RUN git clone https://github.com/ggerganov/llama.cpp.git \
    && cd llama.cpp \
    && cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON \
    && cmake --build build --config Release --target llama-server -j 4

# --- Stage 2: Runtime image nhẹ hơn ---
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy file thực thi từ stage builder sang
COPY --from=builder /app/llama.cpp/build/bin/llama-server /app/llama-server

# Mở cổng API (mặc định của llama.cpp là 8080)
EXPOSE 8300

# Chạy server và chỉ định tải trực tiếp bản Q8 từ Hugging Face thông qua cờ -hf
ENTRYPOINT ["/app/llama-server"]
CMD [ \
    "-hf", "mradermacher/PaddleOCR-VL-1.6-GGUF:Q8_0", \
    "--host", "0.0.0.0", \
    "--port", "8300", \
    "-ngl", "99" \
    ]