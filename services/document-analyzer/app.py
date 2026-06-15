from fastapi import FastAPI, UploadFile, File, Form, Header
from typing import Literal, Optional
import fitz 
import cv2
import numpy as np
import os
import json
import base64
import re
import time
from doclayout_yolo import YOLOv10

from coordinate_processor import CoordinateProcessor
from utils import get_extended_bbox
from native_text_analyzer import NativeTextAnalyzer
from minio_service import MinioService
from llm_ocr_service import LlmOcrService
from smart_ocr_service import SmartOcrService
from ws_layout_analysis import router as ws_router, init_services as ws_init_services

# Import cả 2 chiến lược lưu trữ
from image_cropper_s3_storage import ImageCropperS3
from image_cropper_local_storage import ImageCropperLocal

app = FastAPI(title="Layout Analysis API - Multi-Storage")
app.include_router(ws_router)

# Singleton Services
yolo_model = YOLOv10('/app/models/doclayout_yolo.pt')
coord_processor = CoordinateProcessor()
text_analyzer = NativeTextAnalyzer(scale_factor=3.0)
llm_ocr_svc = LlmOcrService()
smart_ocr_svc = SmartOcrService(llm_ocr_svc)

# Khởi tạo đồng thời cả 2 Storage Strategies
minio_svc = MinioService()
cropper_s3 = ImageCropperS3(minio_service=minio_svc)
cropper_local = ImageCropperLocal(output_base_dir="/app/output")

# Chia sẻ singleton với WebSocket router
ws_init_services(
    yolo_model=yolo_model,
    coord_processor=coord_processor,
    text_analyzer=text_analyzer,
    minio_svc=minio_svc,
    llm_ocr_svc=llm_ocr_svc,
    smart_ocr_svc=smart_ocr_svc,
    cropper_s3=cropper_s3,
    cropper_local=cropper_local,
)

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
    check_digital_text: bool = Form(True),
    doc_recognizer: bool = Form(True),
    table_recognizer: bool = Form(True),
    smart_ocr: bool = Form(True),
    extend_border: Optional[str] = Form(None)
):
    if doc_recognizer:
        check_digital_text = True

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
            
            # 1. Vision & Pre-processing (YOLO Layout Detection)
            start_layout = time.time()
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
            layout_time = time.time() - start_layout
            print(f"[Page {page_idx}] Layout detection took {layout_time:.2f}s (found {len(raw_elements)} boxes)")
                
            # 2. Logic (Tọa độ, Native Text & Crop prep)
            start_prep = time.time()
            processed_elements = coord_processor.process(
                raw_elements, sort, remove_page_header, merge_suspicion
            )
            processed_elements = text_analyzer.analyze(page, processed_elements, check_digital_text)
            
            ocr_elements = []
            ocr_crops = []
            if doc_recognizer:
                for element in processed_elements:
                    is_table = (element.get("label") == "Table" or element.get("predicted_tag") == "table")
                    if is_table and not table_recognizer:
                        element["content"] = ""
                        continue
                    
                    if element.get("digital_text") == "false":
                        x_min, y_min, x_max, y_max = element["bbox"]
                        if extend_border:
                            x_min, y_min, x_max, y_max = get_extended_bbox(
                                [x_min, y_min, x_max, y_max], img_bgr.shape[0], img_bgr.shape[1], extend_border
                            )
                        crop_img = img_bgr[y_min:y_max, x_min:x_max]
                        crop_img = smart_ocr_svc.ensure_bgr(crop_img)
                        ocr_elements.append(element)
                        ocr_crops.append(crop_img)
            prep_time = time.time() - start_prep
            print(f"[Page {page_idx}] JSON processing & crop preparation took {prep_time:.2f}s (prepared {len(ocr_crops)} crops for OCR)")
            
            # --- 2.1 OCR / Recognition logic ---
            if doc_recognizer:
                smart_ocr_svc.run_ocr(
                    ocr_elements=ocr_elements,
                    ocr_crops=ocr_crops,
                    smart_ocr=smart_ocr,
                    storage_type=storage_type,
                    page_num=page_idx,
                    request_id=request_id,
                    user_id=x_user_id,
                    conv_id=x_conversation_id,
                    doc_id=x_document_id,
                    minio_svc=minio_svc,
                    is_pdf=doc.is_pdf
                )
            
            # --- 3. ĐỊNH TUYẾN I/O THEO STORAGE_TYPE ---
            if storage_type == "s3":
                final_page_elements = cropper_s3.crop_and_upload(
                    img_matrix=img_bgr,
                    elements=processed_elements,
                    page_num=page_idx,
                    user_id=x_user_id,
                    conv_id=x_conversation_id,
                    doc_id=x_document_id,
                    extend_border=extend_border
                )
            else:
                # Hàm crop_and_save của class Local cũ
                final_page_elements = cropper_local.crop_and_save(
                    img_matrix=img_bgr,
                    elements=processed_elements,
                    request_id=request_id,
                    page_num=page_idx,
                    extend_border=extend_border
                )
            
            # Cleanup output keys
            for item in final_page_elements:
                if not show_height: item.pop("box_height", None)
                if not show_width: item.pop("box_width", None)
                item.pop("estimated_lines", None)
                if not doc_recognizer:
                    item["content"] = ""
                else:
                    if item.get("content") is None:
                        item["content"] = ""
                
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