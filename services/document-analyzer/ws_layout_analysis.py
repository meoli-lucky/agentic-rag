"""
WebSocket endpoint: /ws/document-analyzer

Protocol:
  1. Client connects via WebSocket.
  2. Client sends ONE JSON message with all params and base64-encoded file:
       {
         "file_b64":          "<base64 encoded PDF/image bytes>",
         "filename":           "document.pdf",
         "x_user_id":          "...",
         "x_conversation_id":  "...",
         "x_document_id":      "...",
         "storage_type":       "s3" | "local",         [default: "s3"]
         "confidence_threshold": 0.25,                 [default: 0.25]
         "sort":               "coordinates"|"confidence", [default: "coordinates"]
         "show_height":        false,                  [default: false]
         "show_width":         false,                  [default: false]
         "remove_page_header": false,                  [default: false]
         "merge_suspicion":    true,                   [default: true]
         "check_digital_text": true,                   [default: true]
         "doc_recognizer":     true,                   [default: true]
         "table_recognizer":   true,                   [default: true]
         "smart_ocr":          true,                   [default: true]
         "crop_padding":       null                    [default: null]
       }
  3. Server streams JSON progress events:
       { "event": "progress", "step": <int>, "step_name": "<str>",
         "page": <int|null>, "total_pages": <int|null>,
         "message": "<str>", "data": <dict|null> }
  4. On completion the server sends:
       { "event": "complete", "result": <full JSON payload> }
  5. On error the server sends:
       { "event": "error", "message": "<str>" }
"""

import asyncio
import base64
import json
import os
import time
import uuid

import cv2
import fitz
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from utils import get_extended_bbox


# ─── Shared service references (populated by app.py via init_services()) ───
_yolo_model        = None
_coord_processor   = None
_text_analyzer     = None
_minio_svc         = None
_llm_ocr_svc       = None
_smart_ocr_svc     = None
_cropper_s3        = None
_cropper_local     = None


def init_services(yolo_model, coord_processor, text_analyzer,
                  minio_svc, llm_ocr_svc, smart_ocr_svc,
                  cropper_s3, cropper_local):
    """Called by app.py after all singletons are ready."""
    global _yolo_model, _coord_processor, _text_analyzer
    global _minio_svc, _llm_ocr_svc, _smart_ocr_svc
    global _cropper_s3, _cropper_local
    _yolo_model      = yolo_model
    _coord_processor = coord_processor
    _text_analyzer   = text_analyzer
    _minio_svc       = minio_svc
    _llm_ocr_svc     = llm_ocr_svc
    _smart_ocr_svc   = smart_ocr_svc
    _cropper_s3      = cropper_s3
    _cropper_local   = cropper_local


# ─── Router ────────────────────────────────────────────────────────────────
router = APIRouter()

# ─── Helper: gửi progress event ────────────────────────────────────────────
async def _send(ws: WebSocket, event: str, **kwargs):
    """Gửi một JSON frame tới client."""
    payload = {"event": event, **kwargs}
    await ws.send_text(json.dumps(payload, ensure_ascii=False))


async def _progress(ws: WebSocket, step: int, step_name: str, message: str,
                    page: int = None, total_pages: int = None, data: dict = None):
    await _send(ws, "progress",
                step=step, step_name=step_name, message=message,
                page=page, total_pages=total_pages, data=data)


# ─── WebSocket endpoint ─────────────────────────────────────────────────────
@router.websocket("/ws/document-analyzer")
async def ws_layout_analysis(websocket: WebSocket):
    await websocket.accept()

    try:
        # ── Nhận tham số từ client ─────────────────────────────────────────
        raw = await websocket.receive_text()
        params = json.loads(raw)

        # Decode file
        file_b64 = params.get("file_b64", "")
        if not file_b64:
            await _send(websocket, "error", message="Missing 'file_b64' field.")
            return
        pdf_bytes = base64.b64decode(file_b64)

        # Parse params với giá trị mặc định
        x_user_id         = params.get("x_user_id", "")
        x_conversation_id = params.get("x_conversation_id", "")
        x_document_id     = params.get("x_document_id", "")
        storage_type          = params.get("storage_type", "s3")
        confidence_threshold  = float(params.get("confidence_threshold", 0.25))
        sort                  = params.get("sort", "coordinates")
        show_height           = bool(params.get("show_height", False))
        show_width            = bool(params.get("show_width", False))
        remove_page_header    = bool(params.get("remove_page_header", False))
        merge_suspicion       = bool(params.get("merge_suspicion", True))
        check_digital_text    = bool(params.get("check_digital_text", True))
        doc_recognizer        = bool(params.get("doc_recognizer", True))
        table_recognizer      = bool(params.get("table_recognizer", True))
        smart_ocr             = bool(params.get("smart_ocr", True))
        crop_padding          = params.get("crop_padding", params.get("extend_border", None))

        # ── Bước 0: Normalize params ───────────────────────────────────────
        if doc_recognizer:
            check_digital_text = True

        await _progress(websocket,
                        step=0, step_name="Normalize Params",
                        message=f"doc_recognizer={doc_recognizer}, "
                                f"check_digital_text={check_digital_text}, "
                                f"smart_ocr={smart_ocr}, storage={storage_type}")

        request_id = str(uuid.uuid4())[:8]
        all_detected_data = []

        # ── Mở PDF ────────────────────────────────────────────────────────
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        is_pdf = doc.is_pdf

        await _progress(websocket,
                        step=0, step_name="File Opened",
                        message=f"Opened document: {total_pages} page(s), is_pdf={is_pdf}",
                        total_pages=total_pages,
                        data={"total_pages": total_pages, "is_pdf": is_pdf})

        yolo = _yolo_model

        try:
            for page_num in range(total_pages):
                page_idx = page_num + 1

                # ── Bước 1: Render page → ảnh ─────────────────────────────
                t0 = time.time()
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) if pix.n == 3 else img_array
                render_time = time.time() - t0

                await _progress(websocket,
                                step=1, step_name="Page Render",
                                message=f"Rendered page {page_idx}/{total_pages} in {render_time:.2f}s "
                                        f"(size: {img_bgr.shape[1]}×{img_bgr.shape[0]}px)",
                                page=page_idx, total_pages=total_pages,
                                data={"width": img_bgr.shape[1], "height": img_bgr.shape[0],
                                      "elapsed_s": round(render_time, 3)})

                # ── Bước 2: YOLOv10 Layout Detection (non-blocking) ──────
                t0 = time.time()
                results = await asyncio.to_thread(
                    yolo, img_bgr,
                    imgsz=1024, conf=confidence_threshold, iou=0.45, agnostic_nms=True
                )
                raw_elements = []
                for box in results[0].boxes:
                    bbox_list = box.xyxy[0].tolist()
                    x_min, y_min, x_max, y_max = bbox_list
                    raw_elements.append({
                        "label": yolo.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": [round(x_min), round(y_min), round(x_max), round(y_max)],
                        "box_width": round(x_max - x_min),
                        "box_height": round(y_max - y_min),
                    })
                layout_time = time.time() - t0

                label_summary = {}
                for el in raw_elements:
                    label_summary[el["label"]] = label_summary.get(el["label"], 0) + 1

                await _progress(websocket,
                                step=2, step_name="Layout Detection",
                                message=f"Page {page_idx}: detected {len(raw_elements)} boxes in {layout_time:.2f}s",
                                page=page_idx, total_pages=total_pages,
                                data={"boxes": len(raw_elements), "labels": label_summary,
                                      "elapsed_s": round(layout_time, 3)})

                # ── Bước 3: CoordinateProcessor ──────────────────────────
                t0 = time.time()
                processed_elements = _coord_processor.process(
                    raw_elements, sort, remove_page_header, merge_suspicion
                )
                coord_time = time.time() - t0

                await _progress(websocket,
                                step=3, step_name="Coordinate Processing",
                                message=f"Page {page_idx}: {len(processed_elements)} elements after dedup/sort "
                                        f"in {coord_time:.3f}s",
                                page=page_idx, total_pages=total_pages,
                                data={"elements_after": len(processed_elements),
                                      "elapsed_s": round(coord_time, 4)})

                # ── Bước 4: NativeTextAnalyzer ────────────────────────────
                t0 = time.time()
                processed_elements = _text_analyzer.analyze(page, processed_elements, check_digital_text)
                text_time = time.time() - t0

                digital_count   = sum(1 for e in processed_elements if e.get("digital_text") == "true")
                non_digital_count = sum(1 for e in processed_elements if e.get("digital_text") == "false")
                unverified_count  = sum(1 for e in processed_elements if e.get("digital_text") == "unverified")

                await _progress(websocket,
                                step=4, step_name="Native Text Extraction",
                                message=f"Page {page_idx}: digital={digital_count}, "
                                        f"needs_ocr={non_digital_count}, unverified={unverified_count} "
                                        f"({text_time:.3f}s)",
                                page=page_idx, total_pages=total_pages,
                                data={"digital": digital_count, "needs_ocr": non_digital_count,
                                      "unverified": unverified_count, "elapsed_s": round(text_time, 4)})

                # ── Bước 5: Chuẩn bị danh sách OCR ───────────────────────
                t0 = time.time()
                ocr_elements = []
                ocr_crops = []
                if doc_recognizer:
                    for element in processed_elements:
                        is_table = (element.get("label") == "Table" or
                                    element.get("predicted_tag") == "table")
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
                            crop_img = _smart_ocr_svc.ensure_bgr(crop_img)
                            ocr_elements.append(element)
                            ocr_crops.append(crop_img)
                prep_time = time.time() - t0

                await _progress(websocket,
                                step=5, step_name="OCR Crop Preparation",
                                message=f"Page {page_idx}: prepared {len(ocr_crops)} crops for OCR "
                                        f"in {prep_time:.3f}s",
                                page=page_idx, total_pages=total_pages,
                                data={"ocr_crops": len(ocr_crops), "elapsed_s": round(prep_time, 4)})

                # ── Bước 6: OCR (SmartOcrService) ────────────────────────
                if doc_recognizer and ocr_crops:
                    t0 = time.time()
                    ocr_mode = "smart" if smart_ocr else "regular"

                    if smart_ocr:
                        chunk_size = _smart_ocr_svc.chunk_size
                        total_chunks = (len(ocr_crops) + chunk_size - 1) // chunk_size
                        await _progress(websocket,
                                        step=6, step_name="OCR Started (Smart)",
                                        message=f"Page {page_idx}: Smart OCR — {len(ocr_crops)} crops → "
                                                f"{total_chunks} chunk(s), chunk_size={chunk_size}",
                                        page=page_idx, total_pages=total_pages,
                                        data={"mode": "smart", "crops": len(ocr_crops),
                                              "chunks": total_chunks, "chunk_size": chunk_size})
                    else:
                        await _progress(websocket,
                                        step=6, step_name="OCR Started (Regular)",
                                        message=f"Page {page_idx}: Regular OCR — {len(ocr_crops)} crops, "
                                                f"1 call per crop",
                                        page=page_idx, total_pages=total_pages,
                                        data={"mode": "regular", "crops": len(ocr_crops)})

                    # Chạy OCR trong thread pool — giải phóng event loop cho WS ping
                    await asyncio.to_thread(
                        _smart_ocr_svc.run_ocr,
                        ocr_elements=ocr_elements,
                        ocr_crops=ocr_crops,
                        smart_ocr=smart_ocr,
                        storage_type=storage_type,
                        page_num=page_idx,
                        request_id=request_id,
                        user_id=x_user_id,
                        conv_id=x_conversation_id,
                        doc_id=x_document_id,
                        minio_svc=_minio_svc,
                        is_pdf=is_pdf,
                    )
                    ocr_time = time.time() - t0

                    await _progress(websocket,
                                    step=6, step_name="OCR Complete",
                                    message=f"Page {page_idx}: OCR finished in {ocr_time:.2f}s "
                                            f"({ocr_mode} mode, {len(ocr_crops)} crop(s))",
                                    page=page_idx, total_pages=total_pages,
                                    data={"mode": ocr_mode, "crops": len(ocr_crops),
                                          "elapsed_s": round(ocr_time, 2)})
                elif doc_recognizer and not ocr_crops:
                    await _progress(websocket,
                                    step=6, step_name="OCR Skipped",
                                    message=f"Page {page_idx}: No non-digital crops — OCR skipped",
                                    page=page_idx, total_pages=total_pages)

                # Tính toán lại line_height dựa trên text thực tế sau khi có kết quả OCR/Native Text
                processed_elements = _coord_processor.recalculate_line_heights(processed_elements)

                # ── Bước 7: Crop & Store thumbnails (non-blocking) ────────
                t0 = time.time()
                if storage_type == "s3":
                    final_page_elements = await asyncio.to_thread(
                        _cropper_s3.crop_and_upload,
                        img_matrix=img_bgr,
                        elements=processed_elements,
                        page_num=page_idx,
                        user_id=x_user_id,
                        conv_id=x_conversation_id,
                        doc_id=x_document_id,
                        crop_padding=crop_padding,
                    )
                else:
                    final_page_elements = await asyncio.to_thread(
                        _cropper_local.crop_and_save,
                        img_matrix=img_bgr,
                        elements=processed_elements,
                        request_id=request_id,
                        page_num=page_idx,
                        crop_padding=crop_padding,
                    )
                store_time = time.time() - t0

                await _progress(websocket,
                                step=7, step_name="Thumbnail Storage",
                                message=f"Page {page_idx}: stored {len(final_page_elements)} thumbnails "
                                        f"to {storage_type.upper()} in {store_time:.2f}s",
                                page=page_idx, total_pages=total_pages,
                                data={"stored": len(final_page_elements), "storage": storage_type,
                                      "elapsed_s": round(store_time, 2)})

                # ── Bước 8: Cleanup JSON fields ───────────────────────────
                for item in final_page_elements:
                    if not show_height: item.pop("box_height", None)
                    if not show_width:  item.pop("box_width", None)
                    item.pop("estimated_lines", None)
                    if not doc_recognizer:
                        item["content"] = ""
                    else:
                        if item.get("content") is None:
                            item["content"] = ""

                all_detected_data.extend(final_page_elements)

                await _progress(websocket,
                                step=8, step_name="Page Complete",
                                message=f"Page {page_idx}/{total_pages} fully processed ✓",
                                page=page_idx, total_pages=total_pages,
                                data={"elements_on_page": len(final_page_elements)})

            # ── Bước 9: Lưu JSON kết quả ──────────────────────────────────
            await _progress(websocket,
                            step=9, step_name="Saving Result JSON",
                            message=f"Saving layout_analysis_result.json to {storage_type.upper()}…")

            final_json_payload = {
                "status": "success",
                "document_metadata": {
                    "user_id": x_user_id,
                    "conversation_id": x_conversation_id,
                    "document_id": x_document_id,
                },
                "total_crops": len([e for e in all_detected_data if e.get("file_path") is not None]),
                "data": all_detected_data,
            }

            if storage_type == "s3":
                json_object_path = _minio_svc.build_object_path(
                    x_user_id, x_conversation_id, x_document_id, "layout_analysis_result.json"
                )
                result_file_url = _minio_svc.upload_json(json_object_path, final_json_payload)
            else:
                local_json_dir = f"/app/output/{request_id}"
                os.makedirs(local_json_dir, exist_ok=True)
                local_json_path = os.path.join(local_json_dir, "layout_analysis_result.json")
                with open(local_json_path, "w", encoding="utf-8") as f:
                    json.dump(final_json_payload, f, ensure_ascii=False, indent=4)
                result_file_url = local_json_path

            final_json_payload["result_file_url"] = result_file_url

            await _progress(websocket,
                            step=9, step_name="Result Saved",
                            message=f"Result JSON saved → {result_file_url}",
                            data={"result_file_url": result_file_url,
                                  "total_elements": len(all_detected_data),
                                  "total_crops": final_json_payload["total_crops"]})

            # ── Hoàn thành ─────────────────────────────────────────────────
            await _send(websocket, "complete", result=final_json_payload)

        finally:
            doc.close()

    except WebSocketDisconnect:
        print("[WS] Client disconnected.")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[WS] Error: {e}\n{tb}")
        try:
            await _send(websocket, "error", message=str(e))
        except Exception:
            pass
