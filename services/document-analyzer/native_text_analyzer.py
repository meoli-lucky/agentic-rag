import fitz

class NativeTextAnalyzer:
    def __init__(self, scale_factor=3.0):
        """
        scale_factor: Hệ số quy đổi tọa độ từ ảnh YOLO sang tọa độ gốc của PDF.
        Mặc định là 3.0 vì ảnh YOLO đang được render ở mức Matrix(3, 3).
        """
        self.scale_factor = scale_factor

    def _check_bbox_content(self, page: fitz.Page, yolo_bbox):
        """Hàm private xử lý nội suy tọa độ và lấy text, đồng thời tối ưu hóa tọa độ bó sát chữ"""
        x_min, y_min, x_max, y_max = yolo_bbox
        
        pdf_rect = fitz.Rect(
            x_min / self.scale_factor,
            y_min / self.scale_factor,
            x_max / self.scale_factor,
            y_max / self.scale_factor
        )
        
        text = page.get_textbox(pdf_rect).strip()
        
        if len(text) > 0:
            # Tối ưu hóa tọa độ bó sát văn bản thực tế từ PDF
            blocks = page.get_text("blocks", clip=pdf_rect)
            if blocks:
                x0 = min(b[0] for b in blocks)
                y0 = min(b[1] for b in blocks)
                x1 = max(b[2] for b in blocks)
                y1 = max(b[3] for b in blocks)
                
                tight_bbox = [
                    round(x0 * self.scale_factor),
                    round(y0 * self.scale_factor),
                    round(x1 * self.scale_factor),
                    round(y1 * self.scale_factor)
                ]
                return "true", text, tight_bbox
            return "true", text, yolo_bbox
        return "false", "", yolo_bbox

    def analyze(self, page: fitz.Page, elements, check_enabled: bool):
        """
        Hàm public để làm giàu (enrich) mảng dữ liệu tọa độ với text nguyên bản.
        """
        for element in elements:
            if not check_enabled:
                element["digital_text"] = "unverified"
                element["content"] = ""
            else:
                is_digital, native_text, tight_bbox = self._check_bbox_content(page, element["bbox"])
                element["digital_text"] = is_digital
                element["content"] = native_text if native_text else ""
                
                # Nếu là digital text, cập nhật lại toạ độ bbox bó sát chữ thực tế để tính toán line_height chuẩn xác
                if is_digital == "true":
                    element["bbox"] = tight_bbox
                    element["box_width"] = tight_bbox[2] - tight_bbox[0]
                    element["box_height"] = tight_bbox[3] - tight_bbox[1]
                
        return elements