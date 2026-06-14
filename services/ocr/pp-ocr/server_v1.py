import os
import shutil
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from paddleocr import PaddleOCR

ocr_instance = None

# --- ĐỌC THAM SỐ CẤU HÌNH ĐỘNG TỪ BÊN NGOÀI ---
USE_ANGLE_CLS = os.getenv("OCR_USE_ANGLE_CLS", "false").lower() == "true"
OCR_LANG = os.getenv("OCR_LANG", "vi")  # 'latin' tối ưu cho Tiếng Việt v4, 'ch' tối ưu cho v5
GPU_ID = int(os.getenv("OCR_GPU_ID", "1"))
OCR_VERSION = os.getenv("OCR_VERSION", "PP-OCRv4") # Cho phép bạn đổi v4 hoặc v5 từ ngoài cờ lệnh

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
    print(f" - GPU Card vật lý: {GPU_ID}")
    print(f" - Ngôn ngữ: {OCR_LANG}")
    print(f" - Tiến trình PID: {os.getpid()}")
    print("==================================================")

    # Khởi tạo bộ tham số cơ bản theo tài liệu quickstart
    params = {
        "ocr_version": OCR_VERSION,
        "det": False,               # Luôn tắt det vì nhận diện ảnh crop từ MinerU
        "use_angle_cls": USE_ANGLE_CLS,
        "lang": OCR_LANG,
        "show_log": False
    }

    # BÀN VÁ TRIỆT ĐỂ LỖI THAM SỐ: Tự chọn cách gọi thiết bị phần cứng theo phiên bản
    if "v5" in OCR_VERSION.lower():
        params["device"] = "gpu"    # Định dạng tham số của v5 / PaddleX core
    else:
        params["use_gpu"] = True    # Định dạng chuẩn v4 theo tài liệu quickstart của bạn
        params["gpu_id"] = 0        # Khi dùng CUDA_VISIBLE_DEVICES, card map vào luôn là ID 0

    ocr_instance = PaddleOCR(**params)
    print(f"✨ Worker [{os.getpid()}] nạp thành công {OCR_VERSION} lên GPU!")
    yield
    print(f"🛑 Worker [{os.getpid()}] đang đóng...")

app = FastAPI(
    title="Production Parallel PaddleOCR Service", 
    version="4.5",
    lifespan=lifespan
)

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

@app.post("/fdsai/api/ocr/paddleocr-v4", response_model=OCRResponse)
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