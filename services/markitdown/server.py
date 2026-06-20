import io
import os
import unicodedata
import re
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from markitdown import MarkItDown
from openai import OpenAI

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MarkItDown Full Feature API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo OpenAI client (nếu có API Key để dùng Vision cho OCR)
client = None
if os.getenv("OPENAI_API_KEY"):
    client = OpenAI()

# CHÚ Ý: Chỉ cần enable_plugins=True, plugin OCR sẽ tự được nạp 
# nếu gói markitdown-ocr đã được cài đặt trong môi trường.
md = MarkItDown(
    enable_plugins=True, 
    llm_client=client,
    llm_model="gpt-4o"
)
def sanitize_filename(filename: str) -> str:
    # 1. Chuẩn hóa Unicode và tách các dấu ra khỏi chữ cái (NFD)
    # Ví dụ: 'ộ' sẽ được tách thành 'o' + dấu mũ + dấu nặng
    temp_name = unicodedata.normalize('NFD', filename)
    
    # 2. Giữ lại các ký tự ASCII (loại bỏ các dấu đã tách ở bước 1)
    # 'ignore' sẽ bỏ qua các ký tự không thể chuyển sang ASCII
    ascii_name = temp_name.encode('ascii', 'ignore').decode('ascii')
    
    # 3. Thay thế khoảng trắng bằng dấu gạch dưới và xóa ký tự lạ (chỉ giữ lại chữ, số, dấu chấm, gạch ngang)
    ascii_name = re.sub(r'[^\w\s.-]', '', ascii_name) # Xóa ký tự đặc biệt
    ascii_name = re.sub(r'\s+', '_', ascii_name)      # Thay khoảng trắng bằng _
    
    return ascii_name

@app.post("/convert")
async def convert_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        file_stream = io.BytesIO(content)
        file_ext = f".{file.filename.split('.')[-1]}"
        
        # Chuyển đổi (OCR sẽ tự chạy ngầm cho ảnh/PDF nếu có llm_client)
        result = md.convert_stream(file_stream, file_extension=file_ext)
        
        return {
            "filename": file.filename,
            "content": result.text_content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export")
async def export_markdown(file: UploadFile = File(...)):
    try:
        content = await file.read()
        result = md.convert_stream(io.BytesIO(content), file_extension=f".{file.filename.split('.')[-1]}")
        
        # Làm sạch tên file trước khi trả về
        raw_base_name = os.path.splitext(file.filename)[0]
        safe_base_name = sanitize_filename(raw_base_name)
        export_filename = f"{safe_base_name}.md"
        
        return Response(
            content=result.text_content,
            media_type="text/markdown",
            headers={
                # Giờ đây tên file chỉ gồm ký tự ASCII, không lo lỗi latin-1
                "Content-Disposition": f"attachment; filename={export_filename}"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)