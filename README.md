# Agentic RAG Services

Workspace chứa các dịch vụ xử lý tài liệu (layout analyzer, OCR, parser, multimodal LLM) phục vụ cho hệ thống RAG nâng cao. Dự án được cấu trúc thành các module độc lập, hỗ trợ cả việc chạy local trực tiếp hoặc đóng gói triển khai bằng Docker/Docker Compose với tăng tốc phần cứng GPU.

---

## 📁 Cấu trúc Thư mục Dự án

```text
agentic-rag/
├── services/
│   └── document-analyzer/       # Dịch vụ phân tích bố cục trang sử dụng DocLayout-YOLO & lưu trữ MinIO
├── vllm-services/
│   ├── mineru/                  # Dịch vụ OCR/Parser tài liệu PDF/ảnh nâng cao dùng MinerU & vLLM
│   ├── paddleocr-vl-1.5/        # Dockerfile chạy PaddleOCR-VL-1.5 bằng vLLM (Port 8100)
│   ├── paddleocr-vl-1.6/        # Dockerfile chạy PaddleOCR-VL-1.6 bằng vLLM (Port 8300)
│   └── qwen2.5-3b/              # Docker Compose chạy song song Qwen2.5-3B và các model VL khác
└── llama-services/
    └── paddle-ocr-vl/           # Dịch vụ chạy PaddleOCR-VL-1.6 GGUF bằng llama.cpp / llama-cpp-python
```

---

## 💻 Hướng dẫn thiết lập Môi trường Local (venv)

Khuyến nghị sử dụng môi trường ảo (`venv`) để tránh xung đột thư viện giữa các dự án.

### 1. Khởi tạo và kích hoạt môi trường ảo (venv)

Chạy các lệnh sau tại thư mục gốc của dự án:

#### Trên Windows (PowerShell / CMD)
* **Khởi tạo**:
  ```powershell
  python -m venv .venv
  ```
* **Kích hoạt (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
  *(Lưu ý: Nếu gặp lỗi quyền thực thi script, hãy chạy lệnh sau: `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process` rồi thử lại).*
* **Kích hoạt (CMD)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```

#### Trên macOS / Ubuntu (Terminal)
* **Khởi tạo**:
  ```bash
  python3 -m venv .venv
  ```
* **Kích hoạt**:
  ```bash
  source .venv/bin/activate
  ```

---

### 2. Cài đặt các thư viện hệ thống bắt buộc (System Dependencies)

#### 🪟 Trên Windows
1. **C++ Build Tools** (Để biên dịch `llama-cpp-python` nếu dùng):
   * Tải và cài đặt [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). Chọn **Desktop development with C++**.
2. **Poppler** (Yêu cầu bởi `pdf2image` phục vụ cho MinerU):
   * Tải bản Poppler mới nhất cho Windows từ [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases).
   * Giải nén và thêm thư mục `bin` vào biến môi trường `PATH` của Windows.

#### 🍎 Trên macOS
Sử dụng `Homebrew` để cài đặt các gói cần thiết:
```bash
# Cài đặt Poppler cho pdf2image
brew install poppler

# Cài đặt compiler cho llama-cpp-python (nếu chưa cài Xcode Command Line Tools)
xcode-select --install
```

#### 🐧 Trên Ubuntu / Debian
```bash
sudo apt update
# Cài đặt Poppler cho pdf2image
sudo apt install -y poppler-utils

# Cài đặt thư viện đồ họa hệ thống cho OpenCV
sudo apt install -y libgl1-mesa-glx libglib2.0-0 libgomp1 libsm6 libxext6 libxrender-dev

# Cài đặt C++ compiler và CMake
sudo apt install -y build-essential cmake
```

---

## 🛠️ Chi tiết cấu hình & Khởi chạy từng Dịch vụ

### 1. Document Analyzer Service (`services/document-analyzer`)
Dịch vụ phân tích bố cục trang tài liệu sử dụng mô hình **DocLayout-YOLO (DocLayNet)**.

* **Cài đặt thư viện (Local)**:
  ```bash
  pip install -r services/document-analyzer/requirements.txt
  ```
* **Khởi chạy Local**:
  ```bash
  uvicorn app:app --host 0.0.0.0 --port 8500 --reload
  ```
* **Chạy bằng Docker Compose** (Khuyên dùng - tự động tích hợp kho lưu trữ MinIO S3):
  ```bash
  cd services/document-analyzer
  docker compose up -d
  ```
  * **API Endpoint**: `http://localhost:8500/api/v1/document-analyzer` (POST)
  * **MinIO API Port**: `9000` (được cấu hình S3 storage cho ảnh crop)
  * **MinIO Console**: `http://localhost:9001` (User: `document_storage` / Pass: `document_storage_password_123`)

---

### 2. MinerU Service (`vllm-services/mineru`)
Dịch vụ phân tích và trích xuất tài liệu (PDF, hình ảnh) sang Markdown nâng cao. Hỗ trợ chạy CLI hoặc chạy dạng API server.

* **Build Docker Images**:
  ```bash
  cd vllm-services/mineru
  # Build bản CLI
  docker build -f Dockerfile-Cli --no-cache -t mineru-cli:v1 .
  # Build bản API Service
  docker build --no-cache -t mineru-api-service:v1 .
  ```

* **Chạy CLI Test**:
  ```bash
  docker run --rm --gpus '"device=1"' --shm-size=2gb -v "$(pwd)":/workspace -v ./magic-pdf.json:/root/magic-pdf.json -w /workspace mineru-cli:v1 mineru -p 742.pdf -o .
  ```

* **Khởi chạy API Service đơn lẻ**:
  ```bash
  docker run -d --name mineru-api-container --gpus '"device=1"' --shm-size=2gb -p 8100:8100 \
    -v "$(pwd)":/workspace \
    -v ./magic-pdf.json:/root/magic-pdf.json \
    -e MINERU_MODEL_SOURCE=local \
    -e VLLM_GPU_MEMORY_UTILIZATION=0.5 \
    -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
    --ipc=host --restart always mineru-api-service:v1
  ```

* **Khởi chạy bằng Docker Compose (Đa chức năng)**:
  Sử dụng file `compose.yaml` hỗ trợ các profiles khác nhau (`openai-server`, `api`, `router`, `gradio`):
  ```bash
  # Ví dụ chạy MinerU API và UI Gradio song song:
  docker compose --profile api --profile gradio up -d
  ```
  * **OpenAI-compatible Server**: Port `30000`
  * **API Service**: Port `8000` (hoặc `8100` tùy theo config Dockerfile/Command)
  * **Router**: Port `8002`
  * **UI Gradio**: `http://localhost:7860`

---

### 3. PaddleOCR-VL Services (`vllm-services/paddleocr-vl-1.5` & `1.6`)
Các Dockerfile đóng gói sẵn môi trường chạy mô hình Vision-Language **PaddleOCR-VL** qua vLLM giúp tối ưu hóa suy luận (Inference).

* **PaddleOCR-VL-1.5** (Chạy trên Port `8100` của container):
  ```bash
  cd vllm-services/paddleocr-vl-1.5
  docker build -t paddleocr-vl-1.5:latest .
  ```
* **PaddleOCR-VL-1.6** (Chạy trên Port `8300` của container):
  ```bash
  cd vllm-services/paddleocr-vl-1.6
  docker build -t paddleocr-vl-1.6:latest .
  ```

---

### 4. Qwen2.5-3B & LLM Orchestration (`vllm-services/qwen2.5-3b`)
Hỗ trợ chạy điều phối đồng thời mô hình ngôn ngữ lớn (LLM) và các dịch vụ OCR/VL.

* **Cài đặt NVIDIA Container Toolkit** (Bắt buộc đối với máy chủ Linux chạy GPU):
  Chi tiết các lệnh cấu hình Docker runtime và sinh CDI có sẵn trong file [cmd.md](file:///e:/Work/Projects/agentic-rag/vllm-services/qwen2.5-3b/cmd.md).
* **Khởi chạy bằng Docker Compose**:
  * **Bản mặc định** (Qwen2.5-3B-Instruct trên GPU 0, PaddleOCR-VL-1.5 trên GPU 1):
    ```bash
    cd vllm-services/qwen2.5-3b
    docker compose up -d
    ```
  * **Bản v1** (Qwen2.5-3B-Instruct trên GPU 0, MinerU2.5-Pro-2604-1.2B trên GPU 1):
    ```bash
    docker compose -f docker-compose_v1.yml up -d
    ```

---

### 5. Llama-cpp Paddle OCR VL (`llama-services/paddle-ocr-vl`)
Dành cho các môi trường tài nguyên hạn chế hoặc muốn sử dụng định dạng GGUF của PaddleOCR-VL-1.6 thông qua `llama.cpp` hoặc `llama-cpp-python`.

* **Chu bị**: Tải mô hình `PaddleOCR-VL-1.6.Q8_0.gguf` và đặt vào thư mục `llama-services/paddle-ocr-vl/models/`.
* **Cách 1: Chạy qua llama-cpp-python API Server (Port 8300)**:
  ```bash
  cd llama-services/paddle-ocr-vl
  docker build -t paddle-ocr-vl-cpp:latest .
  docker run -d --name paddle-ocr-vl-cpp-server --gpus all -p 8300:8300 -v ./models:/models paddle-ocr-vl-cpp:latest
  ```
* **Cách 2: Chạy trực tiếp qua llama.cpp Server (Docker Compose - Port 8300)**:
  ```bash
  docker compose up -d
  ```

---

## 🚀 Lệnh Cài nhanh Môi trường Python (Cơ bản)
Nếu bạn muốn phát triển hoặc test nhanh một vài hàm python tại máy local:
```bash
pip install fastapi uvicorn[standard] python-multipart pymupdf doclayout-yolo huggingface-hub minio==7.2.5 opencv-python pillow pdf2image "mineru[core]>=3.2.1"
```
