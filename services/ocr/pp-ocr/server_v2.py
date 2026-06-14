import os
import sys
import shutil
import asyncio
import subprocess
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from paddleocr import PaddleOCR

# =========================================================================
# 🔥 BẢN VÁ KHỦNG LONG: TỰ ĐỘNG CÀI VÀ LIÊN KẾT CUDNN VÀO TRONG PADDLE LIBS
# =========================================================================
def patch_nvidia_libraries():
    """
    Tự động kiểm tra và cài đặt gói nvidia-cudnn trực tiếp vào môi trường Python.
    Sau đó liên kết mềm toàn bộ file .so vào thư mục core của PaddlePaddle.
    """
    paddle_libs_path = "/usr/local/lib/python3.10/dist-packages/paddle/libs"
    
    # Nếu trong thư mục libs của Paddle đã có sẵn libcudnn thì bỏ qua không cài lại
    if os.path.exists(os.path.join(paddle_libs_path, "libcudnn.so.8")):
        print("✅ Thư viện cuDNN đã được liên kết hoàn chỉnh từ trước.")
        return

    print("📢 Phát hiện thiếu cuDNN runtime. Tiến hành tự động nạp gói cắm nóng (Hot-plug)...")
    try:
        # Tự cài package wheel siêu nhẹ của NVIDIA chứa đầy đủ toán tử .so
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "nvidia-cudnn-cu11==8.9.6.50"],
            check=True
        )
        
        # Tìm xem vị trí package vừa cài nằm ở đâu trong site-packages
        import nvidia.cudnn
        cudnn_lib_dir = os.path.join(os.path.dirname(nvidia.cudnn.__file__), "lib")
        
        if os.path.exists(cudnn_lib_dir):
            os.makedirs(paddle_libs_path, exist_ok=True)
            # Quét và tạo liên kết mềm (symlink) toàn bộ file .so vào thẳng mục đích của Paddle
            for filename in os.listdir(cudnn_lib_dir):
                if filename.endswith(".so") or ".so." in filename:
                    src_file = os.path.join(cudnn_lib_dir, filename)
                    dst_file = os.path.join(paddle_libs_path, filename)
                    if not os.path.exists(dst_file):
                        os.symlink(src_file, dst_file)
            
            # Tạo thêm file alias libcudnn.so từ bản .8 để core C++ nhận diện
            alias_src = os.path.join(paddle_libs_path, "libcudnn.so.8")
            alias_dst = os.path.join(paddle_libs_path, "libcudnn.so")
            if os.path.exists(alias_src) and not os.path.exists(alias_dst):
                os.symlink(alias_src, alias_dst)
                
            print("✨ Đã cài đặt và ánh xạ thành công toàn bộ toán tử cuDNN v8 vào Paddle!")
    except Exception as e:
        print(f"⚠️ Lỗi trong quá trình vá hệ thống: {str(e)}")
        print("Fallback: Ép kích hoạt cờ tắt toán tử cuDNN.")
        os.environ["FLAGS_use_cudnn"] = "0"

# Kích hoạt bản vá lập tức khi container vừa bốc file code lên
patch_nvidia_libraries()

# --- GIỮ NGUYÊN TOÀN BỘ PHẦN CODE KHỞI TẠO VÀ ENDPOINT PHÍA DƯỚI ---
ocr_instance = None
USE_ANGLE_CLS = os.getenv("OCR_USE_ANGLE_CLS", "false").lower() == "true"
OCR_LANG = os.getenv("OCR_LANG", "vi") 
GPU_ID = int(os.getenv("OCR_GPU_ID", "1"))
OCR_VERSION = os.getenv("OCR_VERSION", "PP-OCRv4")

ALLOWED_EXTENSIONS = tuple(
    ext.strip().lower() 
    for ext in os.getenv("OCR_ALLOWED_EXTENSIONS", ".png,.jpg,.jpeg,.tiff,.bmp,.webp").split(",")
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr_instance
    print("==================================================")
    print(f">>> KHỞI TẠO PADDLEOCR PRODUCTION SERVICE <<<")
    print(f" - Phiên bản cấu hình: {OCR_VERSION}")
    print(f" - Ngôn ngữ: {OCR_LANG}")
    print(f" - Tiến trình PID: {os.getpid()}")
    print("==================================================")

    params = {
        "ocr_version": OCR_VERSION,
        "det": False,               
        "use_angle_cls": USE_ANGLE_CLS,
        "lang": OCR_LANG,
        "show_log": False
    }

    if "v5" in OCR_VERSION.lower():
        params["device"] = "gpu"    
    else:
        params["use_gpu"] = True    
        params["gpu_id"] = 0        

    ocr_instance = PaddleOCR(**params)
    print(f"✨ Worker [{os.getpid()}] nạp thành công {OCR_VERSION} lên GPU!")
    yield
    print(f"🛑 Worker [{os.getpid()}] đang đóng...")

app = FastAPI(title="Production Parallel PaddleOCR Service", version="4.5", lifespan=lifespan)

class OCRResponse(BaseModel):
    status: str
    text: str

def run_paddle_inference(image_path: str) -> str:
    global ocr_instance
    result = ocr_instance.ocr(image_path, cls=USE_ANGLE_CLS)
    if not result:
        return ""
    try:
        if isinstance(result[0], tuple):
            texts = [line[0] for line in result if line]
        else:
            texts = [line[0] for line in result[0] if line]
        return " ".join(texts)
    except Exception:
        return ""

@app.post("/api/ocr/pp-ocr", response_model=OCRResponse)
async def run_parallel_ocr(file: UploadFile = File(...)):
    filename = file.filename.lower()
    if not filename.endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Định dạng file không hợp lệ.")

    temp_dir = f"temp_ocr_{asyncio.current_task().get_name()}_{os.getpid()}"
    os.makedirs(temp_dir, exist_ok=True)
    local_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(local_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        loop = asyncio.get_running_loop()
        extracted_text = await loop.run_in_executor(None, run_paddle_inference, local_file_path)
        return OCRResponse(status="success", text=extracted_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)