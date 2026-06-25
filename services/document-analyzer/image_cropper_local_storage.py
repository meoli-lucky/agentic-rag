import os
import cv2
from utils import get_extended_bbox

class ImageCropperLocal:
    def __init__(self, output_base_dir="/app/output"):
        self.output_base_dir = output_base_dir
        os.makedirs(self.output_base_dir, exist_ok=True)

    def crop_and_save(self, img_matrix, elements, request_id, page_num, crop_padding=None):
        """
        Cắt ảnh dựa trên mảng tọa độ và trả về mảng đã được đính kèm đường dẫn file
        """
        current_output_dir = os.path.join(self.output_base_dir, request_id)
        os.makedirs(current_output_dir, exist_ok=True)
        
        final_elements = []
        img_h, img_w = img_matrix.shape[:2]
        
        for index, element in enumerate(elements):
            x_min, y_min, x_max, y_max = element["bbox"]
            
            # Áp dụng mở rộng biên nếu có tùy chọn
            if crop_padding:
                x_min, y_min, x_max, y_max = get_extended_bbox([x_min, y_min, x_max, y_max], img_h, img_w, crop_padding)
                
            lbl = element["label"]
            
            # Cắt ma trận Numpy bằng Slicing (cực nhanh)
            crop_img = img_matrix[y_min:y_max, x_min:x_max]
            
            # Đặt tên và lưu file
            filename = f"page{page_num}_index{index}_{lbl}.jpg"
            file_path = os.path.join(current_output_dir, filename)
            cv2.imwrite(file_path, crop_img)
            
            # Clone dữ liệu và thêm đường dẫn file
            final_item = {
                "page": page_num,
                "index": index,
                **element,  # Giải nén toàn bộ dictionary cũ
                "file_path": file_path
            }
            final_elements.append(final_item)
            
        return final_elements