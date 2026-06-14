import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from paddleocr import PaddleOCR

app = FastAPI(title="PP-OCRv5 Test Service - V3 Pipeline")

print("Đang tải mô hình PP-OCRv5_server (GPU)...")
# 1. Khởi tạo chuẩn V3: Khai báo tham số xoay chữ ở đây thay vì ở hàm predict
 
ocr = PaddleOCR(
    #text_detection_model_name="PP-OCRv5_server_det",
    #text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
    lang="vi",
    use_textline_orientation=False, # Tương đương với cls=True ở bản cũ
    device="gpu:1"
)
print("Tải mô hình thành công!")

@app.post("/api/ocr/pp-ocr")
async def predict_ocr(file: UploadFile = File(...)):
    try:
        # Đọc ảnh từ request
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Không thể đọc hoặc giải mã ảnh.")

        # 2. Gọi hàm predict của V3 (chỉ truyền ảnh, không truyền 'cls')
        results = ocr.predict(img)

        # 3. Trích xuất raw output (Bản V3 hỗ trợ xuất thẳng ra dict JSON rất tiện)
        raw_data = {}
        if results and len(results) > 0:
            # results[0] là kết quả của trang đầu tiên (vì mình gửi 1 ảnh đơn)
            # Thuộc tính .json chứa toàn bộ thông tin gốc do PaddleOCR bóc tách
            raw_data = results[0].json

        return {
            "status": "success",
            "filename": file.filename,
            "raw_output": raw_data
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)