curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml


sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker


docker run --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model Qwen/Qwen3.5-9B --tensor-parallel-size 2 --max-model-len 8192
docker run --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model Qwen/Qwen3.5-9B --tensor-parallel-size 2 --max-model-len 4096 --max-num-seqs 64 --gpu-memory-utilization 0.9 
--enforce-eager

docker run --rm --gpus '"device=0"'   -v ~/.cache/huggingface:/root/.cache/huggingface   -p 8000:8000   --ipc=host   vllm/vllm-openai:latest Qwen/Qwen2.5-3B-Instruct --max-model-len 4096   --max-num-seqs 2   --gpu-memory-utilization 0.75   --dtype float16


docker run -d --name mineru2.5-pro --gpus '"device=1"' -v ~/.cache/huggingface:/root/.cache/huggingface -p 8001:8000 --ipc=host vllm/vllm-openai:latest opendatalab/MinerU2.5-Pro-2604-1.2B --max-model-len 1024 --max-num-seqs 1   --gpu-memory-utilization 0.70  --dtype float16


docker run --rm --name paddleocr-vl --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface -v /opt/ocr-data:/data -p 8100:8100 --ipc=host   paddleocr-vl-1.5:latest