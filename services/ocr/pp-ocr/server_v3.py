import os
import sys
import shutil
import asyncio

# =========================================================================
# 🔥 SỬA CHUẨN CSDN: ÉP CƠ CHẾ SPAWN TIẾN TRÌNH TRƯỚC KHI IMPORT PADDLE
# =========================================================================
import multiprocessing
if __name__ == '__main__' or __name__.startswith('fastapi') or 'uvicorn' in sys.argv[0]:
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("🚀 Đã cấu hình thành công phương thức multiprocessing: SPAWN")
    except RuntimeError:
        pass

# Thêm biến môi trường cô lập tính toán đồ họa đồ thị tĩnh
os.environ["FLAGS_allocator_strategy"] = "auto_growth" # Cấp phát VRAM động theo nhu cầu thực tế

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from paddleocr import PaddleOCR

ocr_instance = None

USE_ANGLE_CLS = os.getenv("OCR_USE_ANGLE_CLS", "false").lower() == "true"
OCR_LANG = os.getenv("OCR_LANG", "vi")  
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
    print(f" - Tiến trình PID độc lập (Spawn): {os.getpid()}")
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

    # Nhờ cơ chế spawn, mỗi worker sẽ tự tạo ra một CUDA Context xịn riêng biệt tại đây
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