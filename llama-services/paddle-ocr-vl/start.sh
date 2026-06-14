#!/bin/bash

# Tham số cấu hình cơ bản:
# --n_gpu_layers -1 : Đẩy toàn bộ model lên VRAM GPU
# --n_ctx 4096      : Cửa sổ ngữ cảnh (tăng lên nếu văn bản OCR xuất ra dài)
# --host 0.0.0.0    : Lắng nghe mọi IP

python -m llama_cpp.server \
    --model /models/PaddleOCR-VL-1.6.Q8_0.gguf \
    --host $HOST \
    --port $PORT \
    --n_gpu_layers -1 \
    --n_ctx 4096 \
    --chat_format llava-1-5 

# LƯU Ý: Nếu model VL của bạn có file Multimodal Projector (mmproj) riêng lẻ, 
# hãy bổ sung thêm tham số: --clip_model_path /models/mmproj-model.gguf