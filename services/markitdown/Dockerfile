# Sử dụng Python 3.12
FROM python:3.12-slim

# Cài đặt các thư viện hệ thống (Dependencies cho PDF, Audio, Office)
RUN apt-get update && apt-get install -y \
    libmagic1 \
    ffmpeg \
    poppler-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài đặt các thư viện Python
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    "markitdown[all]" \
    "markitdown-ocr" \
    openai

# Copy code server vào container
COPY server.py .

# Mở port cho API
EXPOSE 8000

# Chạy server
CMD ["python", "server.py"]