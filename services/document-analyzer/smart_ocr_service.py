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

    def create_divider_image(self, width, height=40):
        divider = np.ones((height, width, 3), dtype=np.uint8) * 255
        text = "[$$$$$]"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        cv2.putText(divider, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        return divider

    def split_ocr_text(self, ocr_text, num_expected):
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
                stacked_list = []
                for idx, img in enumerate(crops_chunk):
                    padded_img = self.pad_image_to_width(img, max_width)
                    stacked_list.append(padded_img)
                    if idx < len(crops_chunk) - 1:
                        stacked_list.append(self.create_divider_image(max_width))
                
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
                    
                    splits = self.split_ocr_text(ocr_text, len(elements_chunk))
                    for el, text_seg in zip(elements_chunk, splits):
                        el["content"] = text_seg
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
