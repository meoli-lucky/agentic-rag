import os
import sys
import shutil
import asyncio
import multiprocessing

# Ép cơ chế SPAWN tiến trình đồng bộ theo chuẩn CSDN để tránh deadlock GPU
if __name__ == '__main__' or __name__.startswith('fastapi') or 'uvicorn' in sys.argv[0]:
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("🚀 Đã cấu hình thành công phương thức multiprocessing: SPAWN")
    except RuntimeError:
        pass

# Cấp phát VRAM động theo nhu cầu thực tế
os.environ["FLAGS_allocator_strategy"] = "auto_growth"

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Any
from paddleocr import PaddleOCR

ocr_instance = None

USE_ANGLE_CLS = os.getenv("OCR_USE_ANGLE_CLS", "false").lower() == "true"
OCR_LANG = os.getenv("OCR_LANG", "en")  
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
    print(f" - Ngôn ngữ Core: {OCR_LANG}")
    print(f" - Tiến trình PID: {os.getpid()}")
    print("==================================================")

    params = {
        "ocr_version": OCR_VERSION,
        "det": False,               # Tắt det cho ảnh crop từ MinerU
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

class OCRRawResponse(BaseModel):
    status: str
    raw_result: Any  # Trả ra mảng ma trận thô, chấp nhận mọi định dạng list/tuple từ Paddle

@app.post("/api/ocr/pp-ocr", response_model=OCRRawResponse)
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
        
        # Đẩy luồng tính toán đồng bộ của PaddleOCR vào Executor độc lập để uvicorn không bị chặn
        loop = asyncio.get_running_loop()
        raw_paddle_output = await loop.run_in_executor(
            None, 
            ocr_instance.ocr, 
            local_file_path, 
            USE_ANGLE_CLS
        )
        
        return OCRRawResponse(status="success", raw_result=raw_paddle_output)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)