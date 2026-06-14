import fitz

class NativeTextAnalyzer:
    def __init__(self, scale_factor=3.0):
        """
        scale_factor: Hệ số quy đổi tọa độ từ ảnh YOLO sang tọa độ gốc của PDF.
        Mặc định là 3.0 vì ảnh YOLO đang được render ở mức Matrix(3, 3).
        """
        self.scale_factor = scale_factor

    def _check_bbox_content(self, page: fitz.Page, yolo_bbox):
        """Hàm private xử lý nội suy tọa độ và lấy text"""
        x_min, y_min, x_max, y_max = yolo_bbox
        
        pdf_rect = fitz.Rect(
            x_min / self.scale_factor,
            y_min / self.scale_factor,
            x_max / self.scale_factor,
            y_max / self.scale_factor
        )
        
        text = page.get_textbox(pdf_rect).strip()
        
        if len(text) > 0:
            return "true", text
        return "false", ""

    def analyze(self, page: fitz.Page, elements, check_enabled: bool):
        """
        Hàm public để làm giàu (enrich) mảng dữ liệu tọa độ với text nguyên bản.
        """
        for element in elements:
            if not check_enabled:
                element["digital_text"] = "unverified"
                element["content"] = ""
            else:
                is_digital, native_text = self._check_bbox_content(page, element["bbox"])
                element["digital_text"] = is_digital
                element["content"] = native_text if native_text else ""
                
        return elements