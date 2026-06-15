class CoordinateProcessor:
    def __init__(self, iou_threshold=0.85, y_tolerance=15):
        self.iou_threshold = iou_threshold
        self.y_tolerance = y_tolerance

    def compute_iou(self, box1, box2):
        """Tính toán tỷ lệ chồng lấp (IoU) để phát hiện Identity Crisis"""
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return intersection_area / float(box1_area + box2_area - intersection_area)

    def filter_duplicates(self, elements):
        """Xóa các box chồng lấp (Giữ lại box có confidence cao nhất)"""
        filtered = []
        # Sort ưu tiên confidence cao lên trước
        elements.sort(key=lambda x: x['confidence'], reverse=True)

        for current_item in elements:
            is_duplicate = False
            for kept_item in filtered:
                if self.compute_iou(current_item['bbox'], kept_item['bbox']) > self.iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append(current_item)
        return filtered

    def sort_spatial(self, elements):
        """Thuật toán XY-Cut: Sắp xếp Từ trên xuống dưới, Từ trái qua phải"""
        elements.sort(key=lambda item: item['bbox'][1]) # Sort sơ bộ trục Y
        
        sorted_elements = []
        current_row = []
        
        for item in elements:
            if not current_row:
                current_row.append(item)
            elif abs(item['bbox'][1] - current_row[0]['bbox'][1]) <= self.y_tolerance:
                current_row.append(item) # Cùng nằm trên 1 dòng
            else:
                current_row.sort(key=lambda x: x['bbox'][0]) # Sort trục X cho dòng vừa xong
                sorted_elements.extend(current_row)
                current_row = [item] # Mở dòng mới
                
        if current_row:
            current_row.sort(key=lambda x: x['bbox'][0])
            sorted_elements.extend(current_row)
            
        return sorted_elements

    def calculate_metrics_and_tags(self, elements):
        """Tính toán chiều cao dòng thực tế và dán nhãn Markdown cho TOÀN BỘ box thô"""
        processed = []
        for element in elements:
            w = element["box_width"]
            h = element["box_height"]
            lbl = element["label"]
            
            # Ước tính số dòng
            aspect_ratio = w / h
            est_lines = 1 if (aspect_ratio > 4.0 or h < 35) else max(1, round(h / 22))
            line_height = round(h / est_lines, 2)
            
            # Gán tag
            tag = "paragraph"
            if lbl == "Section-header":
                tag = "h1" if line_height > 35 else ("h2" if line_height > 25 else "h3")
            elif lbl == "List-item": tag = "list-item"
            elif lbl == "Table": tag = "table"
            elif lbl == "Page-header": tag = "ignore_header"
            
            element["predicted_tag"] = tag
            element["line_height"] = line_height
            element["estimated_lines"] = est_lines
            processed.append(element)
            
        return processed

    def merge_duplicates(self, elements, merge_suspicion):
        """
        Lọc box chồng lấp (IoU > threshold). 
        Nếu merge_suspicion=True, lưu box bị loại vào mảng 'suspicion' của box giữ lại.
        """
        filtered = []
        # Ưu tiên confidence cao nhất làm box "Chủ"
        elements.sort(key=lambda x: x['confidence'], reverse=True)

        for current_item in elements:
            is_duplicate = False
            for kept_item in filtered:
                # Nếu phát hiện chồng lấp tọa độ (Identity Crisis)
                if self.compute_iou(current_item['bbox'], kept_item['bbox']) > self.iou_threshold:
                    is_duplicate = True
                    
                    # LOGIC GỘP NGHI VẤN
                    if merge_suspicion:
                        if "suspicion" not in kept_item:
                            kept_item["suspicion"] = []
                            
                        # Nhét thông tin của box thấp điểm hơn vào mảng suspicion
                        kept_item["suspicion"].append({
                            "label": current_item["label"],
                            "predicted_tag": current_item.get("predicted_tag", "unknown"),
                            "confidence": current_item["confidence"]
                        })
                    break
            
            # Nếu là box mới (Không bị đè)
            if not is_duplicate:
                if merge_suspicion:
                    current_item["suspicion"] = [] # Khởi tạo mảng rỗng cho box chuẩn
                filtered.append(current_item)
                
        return filtered

    def process(self, raw_elements, sort_type, remove_header, merge_suspicion):
        """Pipeline xử lý tọa độ chính"""
        # 1. Lọc rác Header (Nếu yêu cầu)
        if remove_header:
            raw_elements = [el for el in raw_elements if el["label"] != "Page-header"]
            
        # 2. Làm giàu dữ liệu (Tính Tags, Line height TRƯỚC khi gộp)
        tagged_elements = self.calculate_metrics_and_tags(raw_elements)
            
        # 3. Gộp các hộp bị Identity Crisis (De-duplication & Suspicion Merge)
        dedup_elements = self.merge_duplicates(tagged_elements, merge_suspicion)
        
        # 4. Sắp xếp không gian
        if sort_type == "coordinates":
            return self.sort_spatial(dedup_elements)
            
        return dedup_elements

    def recalculate_line_heights(self, elements):
        """
        Tính toán lại line_height dựa trên văn bản thực tế sau khi OCR/trích xuất.
        Sử dụng công thức hình học để ước lượng số dòng một cách chuẩn xác.
        Không thay đổi label hay predicted_tag của YOLO.
        """
        import math
        for el in elements:
            content = el.get("content", "")
            char_count = len(content) if content else 0
            
            box_height = el.get("box_height")
            if box_height is None:
                box_height = el["bbox"][3] - el["bbox"][1]
            box_width = el.get("box_width")
            if box_width is None:
                box_width = el["bbox"][2] - el["bbox"][0]
                
            # Ước lượng số dòng dựa trên hình học & số lượng ký tự (bỏ hoàn toàn việc dựa vào \n từ VLM)
            row_count = 1
            if char_count > 0 and box_width > 0:
                row_count = max(1, round(math.sqrt(0.35 * char_count * (box_height / box_width))))
            
            line_height = round(box_height / (3.0 * row_count) - (row_count if row_count > 1 else 0), 2)
            line_height = max(0.0, line_height)
            
            el["line_height"] = line_height
            el["estimated_lines"] = row_count
            
        return elements

    