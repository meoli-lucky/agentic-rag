import cv2
import numpy as np
import base64
import re
import os

class SmartOcrService:
    def __init__(self, llm_ocr_svc):
        self.llm_ocr_svc = llm_ocr_svc

    def ensure_bgr(self, img):
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def pad_image_to_width(self, img, target_width):
        h, w = img.shape[:2]
        if w == target_width:
            return img
        padded = np.ones((h, target_width, 3), dtype=np.uint8) * 255
        padded[0:h, 0:w] = img
        return padded

    def create_divider_image(self, width, bbox, height=60):
        # Tạo ảnh phân tách màu trắng
        divider = np.ones((height, width, 3), dtype=np.uint8) * 255
        x1, y1, x2, y2 = bbox
        text = f"[bbox][{x1},{y1},{x2},{y2}]"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Đảm bảo chữ vừa khít chiều rộng của divider
        if text_size[0] > width - 20:
            font_scale = max(0.4, font_scale * (width - 20) / text_size[0])
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        cv2.putText(divider, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        return divider

    def split_ocr_text_by_bbox(self, ocr_text, elements_chunk):
        # Khởi tạo nội dung trống mặc định cho toàn bộ elements
        for el in elements_chunk:
            el["content"] = ""
            
        if not ocr_text:
            return
            
        # Regex tìm kiếm tag [bbox][x1,y1,x2,y2]
        # Cho phép khoảng trắng linh hoạt đề phòng LLM OCR tự thêm khoảng trắng
        pattern = r'\[\s*bbox\s*\]\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
        
        parts = re.split(pattern, ocr_text)
        # parts sẽ có cấu trúc: [text_truoc_tag_dau_tien, x1, y1, x2, y2, text_1, x1', y1', ...]
        
        bbox_contents = {}
        parsed_ordered_texts = []
        
        first_segment = parts[0].strip()
        
        i = 1
        while i + 4 < len(parts):
            x1 = parts[i].strip()
            y1 = parts[i+1].strip()
            x2 = parts[i+2].strip()
            y2 = parts[i+3].strip()
            text_content = parts[i+4].strip()
            
            bbox_key = f"{x1},{y1},{x2},{y2}"
            bbox_contents[bbox_key] = text_content
            parsed_ordered_texts.append(text_content)
            
            i += 5
            
        # Ánh xạ nội dung về cho từng element
        for idx, el in enumerate(elements_chunk):
            bbox = el.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            key = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            
            if key in bbox_contents:
                el["content"] = bbox_contents[key]
            else:
                # Fallback: Nếu không khớp key tuyệt đối, dùng thứ tự xuất hiện
                # Nếu là phần tử đầu tiên (idx == 0) và có text trước tag đầu tiên, dùng nó
                if idx == 0 and first_segment:
                    el["content"] = first_segment
                else:
                    # Các phần tử tiếp theo ánh xạ theo thứ tự xuất hiện của các phần nội dung đã parsed
                    fallback_idx = idx if not first_segment else idx - 1
                    if 0 <= fallback_idx < len(parsed_ordered_texts):
                        el["content"] = parsed_ordered_texts[fallback_idx]

    def split_ocr_text(self, ocr_text, num_expected):
        # Hàm cũ để giữ tương thích ngược nếu cần gọi ở đâu đó (nhưng chúng ta đã thay thế trong run_ocr)
        if not ocr_text:
            return [""] * num_expected
        pattern = r'\[\s*\$\s*\$\s*\$\s*\$\s*\$\s*\]|\$\s*\$\s*\$\s*\$\s*\$|\[\s*\$\s*\$\s*\$\s*\$\s*\]|\$\s*\$\s*\$\s*\$\s*'
        splits = re.split(pattern, ocr_text)
        splits = [s.strip() for s in splits]
        
        if len(splits) == num_expected:
            return splits
            
        if len(splits) < num_expected:
            splits.extend([""] * (num_expected - len(splits)))
        elif len(splits) > num_expected:
            splits = splits[:num_expected]
            
        return splits

    def run_ocr(self, ocr_elements, ocr_crops, smart_ocr, storage_type, page_num, request_id, user_id, conv_id, doc_id, minio_svc, is_pdf):
        if not ocr_crops:
            return
            
        import time
        if smart_ocr:
            # Nếu là PDF, ghép tất cả ảnh nhỏ trên trang thành 1 ảnh ghép.
            # Nếu là ảnh (không có số trang), gom nhóm 10 ảnh nhỏ làm 1 ảnh ghép.
            chunk_size = len(ocr_crops) if is_pdf else 10
            
            for chunk_idx in range(0, len(ocr_crops), chunk_size):
                crops_chunk = ocr_crops[chunk_idx:chunk_idx + chunk_size]
                elements_chunk = ocr_elements[chunk_idx:chunk_idx + chunk_size]
                
                max_width = max(img.shape[1] for img in crops_chunk)
                # Đảm bảo max_width tối thiểu là 600px để nhãn bbox trên divider hiển thị rõ ràng
                max_width = max(max_width, 600)
                stacked_list = []
                for idx, (img, el) in enumerate(zip(crops_chunk, elements_chunk)):
                    bbox = el.get("bbox", [0, 0, 0, 0])
                    # Chèn divider chứa toạ độ của crop này ngay phía trước nó
                    stacked_list.append(self.create_divider_image(max_width, bbox))
                    
                    padded_img = self.pad_image_to_width(img, max_width)
                    stacked_list.append(padded_img)
                
                stacked_image = np.vstack(stacked_list)
                
                # Định dạng tên file lưu trữ tương ứng
                part_num = (chunk_idx // chunk_size) + 1
                if is_pdf:
                    filename = f"page{page_num}_stacked.jpg"
                else:
                    filename = f"page{page_num}_stacked_part{part_num}.jpg"
                
                success, buffer = cv2.imencode(".jpg", stacked_image)
                if success:
                    image_bytes = buffer.tobytes()
                    if storage_type == "s3" and minio_svc:
                        object_path = minio_svc.build_object_path(user_id, conv_id, doc_id, filename)
                        minio_svc.upload_image_bytes(object_path, image_bytes)
                    else:
                        local_dir = os.path.join("/app/output", request_id)
                        os.makedirs(local_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(local_dir, filename), stacked_image)
                        
                    # Tiến hành OCR trên ảnh ghép này
                    start_time = time.time()
                    base64_str = base64.b64encode(image_bytes).decode('utf-8')
                    ocr_text = self.llm_ocr_svc.request_ocr(base64_str, is_local=False, is_base64=True)
                    print(f"[Smart OCR] Chunk {part_num} (page {page_num}) took {time.time() - start_time:.2f}s, size: {stacked_image.shape}, crops: {len(crops_chunk)}")
                    
                    # Phân tách và gán nội dung dựa trên bbox
                    self.split_ocr_text_by_bbox(ocr_text, elements_chunk)
        else:
            # Luồng cũ: gọi OCR riêng cho từng ảnh nhỏ
            print(f"[Regular OCR] Starting OCR for {len(ocr_crops)} crops sequentially...")
            for idx, (el, crop_img) in enumerate(zip(ocr_elements, ocr_crops)):
                start_time = time.time()
                success, buffer = cv2.imencode(".jpg", crop_img)
                if success:
                    image_bytes = buffer.tobytes()
                    base64_str = base64.b64encode(image_bytes).decode('utf-8')
                    ocr_text = self.llm_ocr_svc.request_ocr(base64_str, is_local=False, is_base64=True)
                    el["content"] = ocr_text if ocr_text else ""
                    print(f"[Regular OCR] Crop {idx+1}/{len(ocr_crops)} took {time.time() - start_time:.2f}s, size: {crop_img.shape}")
