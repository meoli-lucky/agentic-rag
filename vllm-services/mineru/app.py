from fastapi import FastAPI, UploadFile, File, HTTPException
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.rw.AbsReaderWriter import FileReaderWriter
import uvicorn
import io
import base64
from PIL import Image
from pdf2image import convert_from_bytes

app = FastAPI(title="MinerU Advanced Layout & Crop Service")

def convert_image_to_pdf(image_bytes: bytes) -> bytes:
    """Chuyển đổi file ảnh sang PDF dạng bytes trong bộ nhớ"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        pdf_buffer = io.BytesIO()
        image.save(pdf_buffer, format="PDF")
        return pdf_buffer.getvalue()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý ảnh: {str(e)}")

def pil_to_base64(pil_image: Image.Image) -> str:
    """Chuyển đổi đối tượng ảnh Pillow sang chuỗi Base64 để đóng gói vào JSON"""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

@app.post("/detect-layout")
async def detect_layout(file: UploadFile = File(...)):
    filename = file.filename.lower()
    file_bytes = await file.read()
    
    # 1. Chuẩn hóa tất cả Input về dạng PDF bytes
    if filename.endswith(".pdf"):
        pdf_bytes = file_bytes
    elif filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")):
        pdf_bytes = convert_image_to_pdf(file_bytes)
    else:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file PDF hoặc Ảnh.")

    try:
        # Render trang đầu tiên của PDF thành một đối tượng ảnh Pillow để làm gốc Crop tọa độ
        # (MinerU hiện tại phân tích layout theo từng trang, ở đây MVP xử lý trang 1/ảnh đơn)
        pages_images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1)
        if not pages_images:
            raise Exception("Không thể render trang PDF thành hình ảnh.")
        base_page_image = pages_images[0]

        # 2. Khởi tạo và chạy Pipeline MinerU đầy đủ hệ thống
        jso_useful_key = {"_pdf_type": "", "model_list": []}
        pipe = UNIPipe(pdf_bytes, jso_useful_key, FileReaderWriter())

        pipe.parse_layout()              # Quét Layout toàn cục
        pipe.parse_table_and_formula()   # Quét sâu cấu trúc Bảng và Công thức

        formatted_layout = []
        
        # 3. Duyệt qua các block layout được bóc tách
        for idx, block in enumerate(pipe.model_list):
            bbox = block.get("bbox", []) # Tọa độ: [x0, y0, x1, y1]
            category_type = block.get("type", "unknown")
            score = block.get("score", 0.0)
            
            block_data = {
                "block_id": idx,
                "type": category_type,
                "bbox": bbox,
                "confidence": round(score, 4),
                "action": "PADDLE_OCR",
                "table_content": None,
                "cropped_image_base64": None
            }
            
            # Xử lý Rẽ nhánh theo chiến lược Hybrid
            if category_type == "table":
                # Nhánh 1: Nếu là Bảng -> Lấy luôn kết quả cấu trúc text nội bộ của MinerU
                block_data["action"] = "USE_MINERU_RESULT"
                block_data["table_content"] = block.get("html", block.get("markdown", ""))
            else:
                # Nhánh 2: Nếu là phân đoạn chữ thường -> Thực hiện CROP ảnh trực tiếp
                if len(bbox) == 4:
                    try:
                        # Pillow crop nhận tuple: (left, upper, right, lower)
                        cropped_box = base_page_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                        # Đóng gói vùng ảnh vừa cắt thành chuỗi Base64
                        block_data["cropped_image_base64"] = pil_to_base64(cropped_box)
                    except Exception as crop_err:
                        block_data["action"] = "CROP_FAILED"
                        block_data["error"] = str(crop_err)

            formatted_layout.append(block_data)

        return {
            "filename": file.filename,
            "status": "success",
            "layout": formatted_layout
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống pipeline: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8100, reload=False)