from fastapi import FastAPI, UploadFile, File, Form, Header
from typing import Literal, Optional, Annotated
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

def download_file_from_url(url: str) -> bytes:
    import requests
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content

async def process_layout_analysis_pdf_bytes(
    pdf_bytes: bytes,
    x_user_id: str,
    x_conversation_id: str,
    x_document_id: str,
    storage_type: Literal["s3", "local"],
    confidence_threshold: float,
    sort: Literal["confidence", "coordinates"],
    show_height: bool,
    show_width: bool,
    remove_page_header: bool,
    merge_suspicion: bool,
    check_digital_text: bool,
    doc_recognizer: bool,
    table_recognizer: bool,
    smart_ocr: bool,
    crop_padding: Optional[str],
    filename: Optional[str] = None,
    content_type: Optional[str] = None
):
    if doc_recognizer:
        check_digital_text = True

    import uuid
    request_id = str(uuid.uuid4())[:8] # Sinh ID cho local storage
    all_detected_data = []
    
    # Nếu file được upload trực tiếp, lưu lại file nguồn theo storage_type
    if filename:
        if storage_type == "s3":
            object_name = minio_svc.build_object_path(x_user_id, x_conversation_id, x_document_id, filename)
            minio_svc.upload_file_bytes(object_name, pdf_bytes, content_type=content_type or "application/pdf")
        else:
            local_file_dir = f"/app/output/{request_id}"
            os.makedirs(local_file_dir, exist_ok=True)
            local_file_path = os.path.join(local_file_dir, filename)
            with open(local_file_path, "wb") as f:
                f.write(pdf_bytes)
    
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
                        if crop_padding:
                            x_min, y_min, x_max, y_max = get_extended_bbox(
                                [x_min, y_min, x_max, y_max], img_bgr.shape[0], img_bgr.shape[1], crop_padding
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
            
            # Tính toán lại line_height dựa trên text thực tế sau khi có kết quả OCR/Native Text
            processed_elements = coord_processor.recalculate_line_heights(processed_elements)
            
            # --- 3. ĐỊNH TUYẾN I/O THEO STORAGE_TYPE ---
            if storage_type == "s3":
                final_page_elements = cropper_s3.crop_and_upload(
                    img_matrix=img_bgr,
                    elements=processed_elements,
                    page_num=page_idx,
                    user_id=x_user_id,
                    conv_id=x_conversation_id,
                    doc_id=x_document_id,
                    crop_padding=crop_padding
                )
            else:
                # Hàm crop_and_save của class Local cũ
                final_page_elements = cropper_local.crop_and_save(
                    img_matrix=img_bgr,
                    elements=processed_elements,
                    request_id=request_id,
                    page_num=page_idx,
                    crop_padding=crop_padding
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
        
        if filename:
            if storage_type == "s3":
                original_file_url = f"{minio_svc.bucket_name}/{minio_svc.build_object_path(x_user_id, x_conversation_id, x_document_id, filename)}"
            else:
                original_file_url = os.path.join(f"/app/output/{request_id}", filename)
            final_json_payload["original_file_url"] = original_file_url
            
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

@app.post("/api/v1/document-analyzer")
async def layout_analysis(
    file: Annotated[Optional[UploadFile], File()] = None,
    file_url: Annotated[Optional[str], Form(description="Presigned URL của MinIO chứa file PDF")] = None,
    
    # Headers
    x_user_id: str = Header(..., description="UUID của User"),
    x_conversation_id: str = Header(..., description="ID của phiên chat/hồ sơ"),
    x_document_id: str = Header(..., description="ID của tài liệu"),
    
    # --- OPTION MỚI: CHỌN LUỒNG LƯU TRỮ ---
    storage_type: Annotated[Literal["s3", "local"], Form(description="Đích lưu trữ (s3 hoặc local)")] = "s3",
    
    confidence_threshold: Annotated[float, Form()] = 0.25,
    sort: Annotated[Literal["confidence", "coordinates"], Form()] = "coordinates",
    show_height: Annotated[bool, Form()] = False,
    show_width: Annotated[bool, Form()] = False,
    remove_page_header: Annotated[bool, Form()] = False,
    merge_suspicion: Annotated[bool, Form()] = True,
    check_digital_text: Annotated[bool, Form()] = True,
    doc_recognizer: Annotated[bool, Form()] = True,
    table_recognizer: Annotated[bool, Form()] = True,
    smart_ocr: Annotated[bool, Form()] = True,
    crop_padding: Annotated[Optional[str], Form()] = None
):
    if not file and not file_url:
        return {"status": "error", "message": "Either 'file' or 'file_url' must be provided."}
    if file and file_url:
        return {"status": "error", "message": "Only one of 'file' or 'file_url' should be provided."}

    if file:
        pdf_bytes = await file.read()
    else:
        import asyncio
        try:
            pdf_bytes = await asyncio.to_thread(download_file_from_url, file_url)
        except Exception as e:
            return {"status": "error", "message": f"Failed to download file from URL: {e}"}

    return await process_layout_analysis_pdf_bytes(
        pdf_bytes=pdf_bytes,
        x_user_id=x_user_id,
        x_conversation_id=x_conversation_id,
        x_document_id=x_document_id,
        storage_type=storage_type,
        confidence_threshold=confidence_threshold,
        sort=sort,
        show_height=show_height,
        show_width=show_width,
        remove_page_header=remove_page_header,
        merge_suspicion=merge_suspicion,
        check_digital_text=check_digital_text,
        doc_recognizer=doc_recognizer,
        table_recognizer=table_recognizer,
        smart_ocr=smart_ocr,
        crop_padding=crop_padding,
        filename=file.filename if file else None,
        content_type=file.content_type if file else None
    )

# ---------------------------------------------------------
# DỊCH VỤ SEMANTIC SEARCH (Warm Start) TÌM KIẾM QDRANT QUA API
# ---------------------------------------------------------
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Embedding Service] Loading SentenceTransformer on device: {device}", flush=True)
        embedding_model = SentenceTransformer("keepitreal/vietnamese-sbert", device=device)
    return embedding_model

@app.post("/api/v1/semantic-search")
async def semantic_search(
    query: Annotated[str, Form(description="Câu truy vấn ngữ nghĩa")],
    x_user_id: Annotated[str, Header(..., description="UUID của User")],
    x_conversation_id: Annotated[str, Header(..., description="ID của phiên chat/hồ sơ")],
    x_document_id: Annotated[str, Header(..., description="ID của tài liệu")],
    top_k: Annotated[int, Form(description="Số lượng kết quả phù hợp nhất")] = 5,
    threshold: Annotated[float, Form(description="Độ tin cậy tối thiểu")] = 0.0
):
    from qdrant_client import QdrantClient
    
    collection_name = f"{x_user_id}_{x_conversation_id}_{x_document_id}"
    
    try:
        # 1. Sinh embedding cho câu hỏi
        model = get_embedding_model()
        query_vector = model.encode(query).tolist()
        
        # 2. Kết nối tới Qdrant (sử dụng host 'qdrant' trong docker-compose network)
        qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # 3. Thực hiện truy vấn Qdrant bằng API Query mới nhất (.query_points)
        if not client.collection_exists(collection_name):
            return {"status": "success", "results": []}
            
        search_result = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=threshold if threshold > 0.0 else None
        )
        
        results = []
        for hit in search_result.points:
            results.append({
                "id": hit.id,
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "metadata": hit.payload.get("metadata", {}),
                "x_user_id": hit.payload.get("x_user_id", ""),
                "x_conversation_id": hit.payload.get("x_conversation_id", ""),
                "x_document_id": hit.payload.get("x_document_id", "")
            })
            
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/v1/index-document")
async def index_document(
    text: Annotated[str, Form(description="Nội dung văn bản Markdown cần chunking & indexing")],
    x_user_id: Annotated[str, Header(..., description="UUID của User")],
    x_conversation_id: Annotated[str, Header(..., description="ID của phiên chat/hồ sơ")],
    x_document_id: Annotated[str, Header(..., description="ID của tài liệu")]
):
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, VectorParams, Distance
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    
    collection_name = f"{x_user_id}_{x_conversation_id}_{x_document_id}"
    
    try:
        # 1. Chunking theo Header của Markdown
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(text)
        
        if not md_header_splits:
            return {"status": "success", "indexed_points": 0}
            
        # 2. Sinh embedding cho từng chunk bằng warm model
        model = get_embedding_model()
        
        # 3. Kết nối Qdrant
        qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Tạo collection nếu chưa tồn tại
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            
        points = []
        for i, split in enumerate(md_header_splits):
            chunk_text = split.page_content
            # Sinh vector
            vector = model.encode(chunk_text).tolist()
            
            points.append(PointStruct(
                id=i,
                vector=vector,
                payload={
                    "text": chunk_text,
                    "metadata": split.metadata,
                    "x_user_id": x_user_id,
                    "x_conversation_id": x_conversation_id,
                    "x_document_id": x_document_id
                }
            ))
            
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        return {"status": "success", "indexed_points": len(points)}
    except Exception as e:
        return {"status": "error", "message": str(e)}