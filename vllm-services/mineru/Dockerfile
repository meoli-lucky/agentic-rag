# Use the official vllm image for gpu with Volta、Turing、Ampere、Ada Lovelace、Hopper、Blackwell architecture (7.0 <= Compute Capability <= 12.1)
# The default base image uses vLLM 0.21.0 with CUDA 13.0. For CUDA 12.9 environments, switch to the commented cu129 image below.
# Compute Capability version query (https://developer.nvidia.com/cuda-gpus)
# support x86_64 architecture and ARM(AArch64) architecture
FROM vllm/vllm-openai:v0.21.0
# FROM vllm/vllm-openai:v0.21.0-cu129

# Install libgl for opencv support & Noto fonts for Chinese characters
RUN apt-get update && \
    apt-get install -y \
    fonts-noto-core \
    fonts-noto-cjk \
    fontconfig \
    libgl1 && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install mineru latest
RUN python3 -m pip install -U 'mineru[core]>=3.2.1' --break-system-packages && \
    python3 -m pip cache purge

RUN /bin/bash -c "mineru-models-download -s huggingface -m all"

WORKDIR /workspace
ENV MINERU_MODEL_SOURCE=local

EXPOSE 8100

ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && exec \"$@\"", "--"]

# Run mineru fast API
CMD ["python3", "-m", "mineru.cli.fast_api", "--host", "0.0.0.0", "--port", "8100", "--gpu-memory-utilization", "0.5", "--allow-public-http-client", "0.0.0.0"]