FROM vllm/vllm-openai:latest

# Đặt biến môi trường lưu Cache để tận dụng Model đã tải nếu mount volume
ENV HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface

EXPOSE 8300

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

CMD [ \
    "--model", "PaddlePaddle/PaddleOCR-VL-1.6", \
    "--host", "0.0.0.0", \
    "--port", "8300", \
    "--max-model-len", "4096", \
    "--enforce-eager", \
    "--gpu-memory-utilization", "0.8", \
    "--limit-mm-per-prompt", "{\"image\":5}", \
    "--dtype", "float16", \
    "--trust-remote-code" \
    ]