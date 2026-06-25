import cv2
import numpy as np
import base64
import re
import os
from concurrent.futures import ThreadPoolExecutor

class SmartOcrService:
    def __init__(self, llm_ocr_svc):
        self.llm_ocr_svc = llm_ocr_svc
        self.chunk_size = 4
        self.max_threads = int(os.getenv("OCR_MAX_THREADS", "3"))

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

    def create_divider_image(self, width, idx, height=60):
        # Tạo ảnh phân tách màu trắng
        divider = np.ones((height, width, 3), dtype=np.uint8) * 255
        text = f"[box_{idx}]"
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
            
        # 1. Thử khớp theo tag index: [box_N], [box-N], [box N], [boxN] hoặc [box_end]
        index_pattern = r'\[\s*box[_\s-]*(\d+|end)\s*\]'
        if re.search(index_pattern, ocr_text):
            parts = re.split(index_pattern, ocr_text)
            first_segment = parts[0].strip()
            index_contents = {}
            parsed_ordered_texts = []
            
            i = 1
            while i + 1 < len(parts):
                idx_str = parts[i].strip()
                text_content = parts[i+1].strip()
                if idx_str == 'end':
                    i += 2
                    continue
                try:
                    idx_val = int(idx_str)
                    index_contents[idx_val] = text_content
                except ValueError:
                    pass
                parsed_ordered_texts.append(text_content)
                i += 2
                
            for idx, el in enumerate(elements_chunk):
                if idx in index_contents:
                    el["content"] = index_contents[idx]
                else:
                    # Fallback: Dùng thứ tự xuất hiện
                    if idx == 0 and first_segment:
                        el["content"] = first_segment
                    else:
                        fallback_idx = idx if not first_segment else idx - 1
                        if 0 <= fallback_idx < len(parsed_ordered_texts):
                            el["content"] = parsed_ordered_texts[fallback_idx]
            return

        # 2. Thử khớp theo tag tọa độ (tương thích ngược nếu VLM dùng bản cũ hoặc ảnh cũ): [bbox][x1,y1,x2,y2]
        bbox_pattern = r'\[\s*bbox\s*\]\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
        if re.search(bbox_pattern, ocr_text):
            parts = re.split(bbox_pattern, ocr_text)
            first_segment = parts[0].strip()
            bbox_contents = {}
            parsed_ordered_texts = []
            
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
                
            for idx, el in enumerate(elements_chunk):
                bbox = el.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                key = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
                if key in bbox_contents:
                    el["content"] = bbox_contents[key]
                else:
                    if idx == 0 and first_segment:
                        el["content"] = first_segment
                    else:
                        fallback_idx = idx if not first_segment else idx - 1
                        if 0 <= fallback_idx < len(parsed_ordered_texts):
                            el["content"] = parsed_ordered_texts[fallback_idx]
            return

        # 3. Fallback cuối cùng: Tách theo các kí tự phân tách tĩnh như [$$$$$]
        splits = self.split_ocr_text(ocr_text, len(elements_chunk))
        for idx, el in enumerate(elements_chunk):
            if idx < len(splits):
                el["content"] = splits[idx]

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

        max_workers = self.max_threads
        max_crop_width = int(os.getenv("OCR_MAX_CROP_WIDTH", "1600"))
        print(f"[OCR] Starting OCR for page {page_num} with max_threads={max_workers} (smart_ocr={smart_ocr}, max_crop_width={max_crop_width})")

        if smart_ocr:
            # Giới hạn số lượng ảnh ghép mỗi chunk để tránh VLM nén ảnh làm mờ nét chữ
            chunk_size = self.chunk_size
            tasks = []
            
            for chunk_idx in range(0, len(ocr_crops), chunk_size):
                crops_chunk = ocr_crops[chunk_idx:chunk_idx + chunk_size]
                elements_chunk = ocr_elements[chunk_idx:chunk_idx + chunk_size]
                
                # Resize các crop có kích thước quá lớn trước khi xếp chồng lên nhau
                resized_crops = []
                for img in crops_chunk:
                    h, w = img.shape[:2]
                    if w > max_crop_width:
                        new_w = max_crop_width
                        new_h = max(1, int(h * (max_crop_width / w)))
                        print(f"[Smart OCR] Resizing oversized crop from {w}x{h} to {new_w}x{new_h}", flush=True)
                        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    resized_crops.append(img)
                crops_chunk = resized_crops
                
                max_width = max(img.shape[1] for img in crops_chunk)
                # Đảm bảo max_width tối thiểu là 600px để nhãn bbox trên divider hiển thị rõ ràng
                max_width = max(max_width, 600)
                stacked_list = []
                for idx, (img, el) in enumerate(zip(crops_chunk, elements_chunk)):
                    # Chèn divider chứa index của crop này ngay phía trước nó
                    stacked_list.append(self.create_divider_image(max_width, idx))
                    
                    padded_img = self.pad_image_to_width(img, max_width)
                    stacked_list.append(padded_img)
                
                # Chèn thêm một divider kết thúc ở cuối cùng để giữ cấu trúc cho crop cuối
                stacked_list.append(self.create_divider_image(max_width, "end", height=40))
                
                stacked_image = np.vstack(stacked_list)
                
                # Định dạng tên file lưu trữ tương ứng
                part_num = (chunk_idx // chunk_size) + 1
                if is_pdf:
                    # Nếu tổng số crops của trang nhỏ hơn chunk_size thì dùng tên page mặc định,
                    # ngược lại thêm part_num để tránh ghi đè file giữa các chunk của cùng một page.
                    if len(ocr_crops) <= chunk_size:
                        filename = f"page{page_num}_stacked.jpg"
                    else:
                        filename = f"page{page_num}_stacked_part{part_num}.jpg"
                else:
                    filename = f"page{page_num}_stacked_part{part_num}.jpg"
                
                tasks.append({
                    "stacked_image": stacked_image,
                    "elements_chunk": elements_chunk,
                    "crops_chunk": crops_chunk,
                    "filename": filename,
                    "part_num": part_num
                })

            def process_chunk(task):
                try:
                    stacked_image = task["stacked_image"]
                    elements_chunk = task["elements_chunk"]
                    crops_chunk = task["crops_chunk"]
                    filename = task["filename"]
                    part_num = task["part_num"]
                    
                    success, buffer = cv2.imencode(".jpg", stacked_image)
                    if not success:
                        print(f"[Smart OCR] Failed to encode stacked image for chunk {part_num}")
                        return
                    
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
                    h, w = stacked_image.shape[:2]
                    print(f"[Smart OCR] Chunk {part_num} (page {page_num}) took {time.time() - start_time:.2f}s, image size: {w}x{h} (width x height), crops: {len(crops_chunk)}")
                    
                    # Phân tách và gán nội dung dựa trên bbox
                    self.split_ocr_text_by_bbox(ocr_text, elements_chunk)
                except Exception as e:
                    print(f"[Smart OCR] Error processing chunk {task.get('part_num')}: {e}")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(process_chunk, tasks)

        else:
            # Luồng cũ: gọi OCR riêng cho từng ảnh nhỏ
            print(f"[Regular OCR] Starting OCR for {len(ocr_crops)} crops concurrently with {max_workers} threads...")
            tasks = []
            for idx, (el, crop_img) in enumerate(zip(ocr_elements, ocr_crops)):
                # Resize các crop có kích thước quá lớn
                h, w = crop_img.shape[:2]
                if w > max_crop_width:
                    new_w = max_crop_width
                    new_h = max(1, int(h * (max_crop_width / w)))
                    print(f"[Regular OCR] Resizing crop {idx+1} from {w}x{h} to {new_w}x{new_h}", flush=True)
                    crop_img = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                tasks.append({
                    "idx": idx,
                    "el": el,
                    "crop_img": crop_img
                })

            def process_crop(task):
                try:
                    idx = task["idx"]
                    el = task["el"]
                    crop_img = task["crop_img"]
                    
                    success, buffer = cv2.imencode(".jpg", crop_img)
                    if not success:
                        print(f"[Regular OCR] Failed to encode crop {idx+1}")
                        return
                    
                    image_bytes = buffer.tobytes()
                    start_time = time.time()
                    base64_str = base64.b64encode(image_bytes).decode('utf-8')
                    ocr_text = self.llm_ocr_svc.request_ocr(base64_str, is_local=False, is_base64=True)
                    el["content"] = ocr_text if ocr_text else ""
                    print(f"[Regular OCR] Crop {idx+1}/{len(ocr_crops)} took {time.time() - start_time:.2f}s, size: {crop_img.shape}")
                except Exception as e:
                    print(f"[Regular OCR] Error processing crop {task.get('idx')}: {e}")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(process_crop, tasks)
