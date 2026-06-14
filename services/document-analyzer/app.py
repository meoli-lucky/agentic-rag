from fastapi import FastAPI, UploadFile, File, Form, Header
from typing import Literal, Optional
import fitz 
import cv2
import numpy as np
import os
import json
from doclayout_yolo import YOLOv10

from coordinate_processor import CoordinateProcessor
from native_text_analyzer import NativeTextAnalyzer
from minio_service import MinioService

# Import cả 2 chiến lược lưu trữ
from image_cropper_s3_storage import ImageCropperS3
from image_cropper_local_storage import ImageCropperLocal

app = FastAPI(title="Layout Analysis API - Multi-Storage")

# Singleton Services
yolo_model = YOLOv10('/app/models/doclayout_yolo.pt')
coord_processor = CoordinateProcessor()
text_analyzer = NativeTextAnalyzer(scale_factor=3.0)

# Khởi tạo đồng thời cả 2 Storage Strategies
minio_svc = MinioService()
cropper_s3 = ImageCropperS3(minio_service=minio_svc)
cropper_local = ImageCropperLocal(output_base_dir="/app/output")

@app.post("/api/v1/layout-analysis")
async def layout_analysis(
    file: UploadFile = File(...),
    
    # Headers
    x_user_id: str = Header(..., description="UUID của User"),
    x_conversation_id: str = Header(..., description="ID của phiên chat/hồ sơ"),
    x_document_id: str = Header(..., description="ID của tài liệu"),
    
    # --- OPTION MỚI: CHỌN LUỒNG LƯU TRỮ ---
    storage_type: Literal["s3", "local"] = Form("s3", description="Đích lưu trữ (s3 hoặc local)"),
    
    confidence_threshold: float = Form(0.25),
    sort: Literal["confidence", "coordinates"] = Form("coordinates"),
    show_height: bool = Form(False),
    show_width: bool = Form(False),
    remove_page_header: bool = Form(False),
    merge_suspicion: bool = Form(True),
    check_digital_text: bool = Form(True)
):
    import uuid
    request_id = str(uuid.uuid4())[:8] # Sinh ID cho local storage
    pdf_bytes = await file.read()
    all_detected_data = []
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        for page_num in range(len(doc)):
            page_idx = page_num + 1
            page = doc.load_page(page_num)
            
            # 1. Vision & Pre-processing
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) if pix.n == 3 else img_array
            
            results = yolo_model(img_bgr, imgsz=1024, conf=confidence_threshold, iou=0.45, agnostic_nms=True)
            
            raw_elements = []
            for box in results[0].boxes:
                # 1. Ép kiểu Tensor về List chuẩn của Python ngay lập tức
                bbox_list = box.xyxy[0].tolist() 
                x_min, y_min, x_max, y_max = bbox_list
            
                raw_elements.append({
                    "label": yolo_model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": [round(x_min), round(y_min), round(x_max), round(y_max)],
                    "box_width": round(x_max - x_min),
                    "box_height": round(y_max - y_min)
                })
                
            # 2. Logic (Tọa độ & Native Text)
            processed_elements = coord_processor.process(
                raw_elements, sort, remove_page_header, merge_suspicion
            )
            processed_elements = text_analyzer.analyze(page, processed_elements, check_digital_text)
            
            # --- 3. ĐỊNH TUYẾN I/O THEO STORAGE_TYPE ---
            if storage_type == "s3":
                final_page_elements = cropper_s3.crop_and_upload(
                    img_matrix=img_bgr,
                    elements=processed_elements,
                    page_num=page_idx,
                    user_id=x_user_id,
                    conv_id=x_conversation_id,
                    doc_id=x_document_id
                )
            else:
                # Hàm crop_and_save của class Local cũ
                final_page_elements = cropper_local.crop_and_save(
                    img_matrix=img_bgr,
                    elements=processed_elements,
                    request_id=request_id,
                    page_num=page_idx
                )
            
            # Cleanup output keys
            for item in final_page_elements:
                if not show_height: item.pop("box_height", None)
                if not show_width: item.pop("box_width", None)
                item.pop("estimated_lines", None)
                if item.get("extracted_text") is None: item.pop("extracted_text", None)
                
            all_detected_data.extend(final_page_elements)

        # --- 4. LƯU KẾT QUẢ JSON THEO STORAGE_TYPE ---
        final_json_payload = {
            "status": "success",
            "document_metadata": {
                "user_id": x_user_id,
                "conversation_id": x_conversation_id,
                "document_id": x_document_id
            },
            "total_crops": len([e for e in all_detected_data if e.get("file_path") is not None]),
            "data": all_detected_data
        }
        
        if storage_type == "s3":
            json_object_path = minio_svc.build_object_path(x_user_id, x_conversation_id, x_document_id, "layout_analysis_result.json")
            result_file_url = minio_svc.upload_json(json_object_path, final_json_payload)
        else:
            local_json_dir = f"/app/output/{request_id}"
            os.makedirs(local_json_dir, exist_ok=True)
            local_json_path = os.path.join(local_json_dir, "layout_analysis_result.json")
            
            with open(local_json_path, 'w', encoding='utf-8') as f:
                json.dump(final_json_payload, f, ensure_ascii=False, indent=4)
            result_file_url = local_json_path

    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if 'doc' in locals():
            doc.close()
            
    final_json_payload["result_file_url"] = result_file_url
    return final_json_payload