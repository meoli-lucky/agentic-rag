import cv2
from utils import get_extended_bbox

class ImageCropperS3:
    def __init__(self, minio_service):
        self.minio_service = minio_service

    def crop_and_upload(self, img_matrix, elements, page_num, user_id, conv_id, doc_id, extend_border=None):
        """
        Cắt ảnh và upload thẳng lên MinIO, không chạm ổ cứng.
        """
        final_elements = []
        img_h, img_w = img_matrix.shape[:2]
        
        for index, element in enumerate(elements):
            x_min, y_min, x_max, y_max = element["bbox"]
            
            # Áp dụng mở rộng biên nếu có tùy chọn
            if extend_border:
                x_min, y_min, x_max, y_max = get_extended_bbox([x_min, y_min, x_max, y_max], img_h, img_w, extend_border)
                
            lbl = element["label"]
            
            # LUÔN LUÔN CẮT ẢNH CHO MỌI BLOCK (Để phục vụ UI Preview)
            crop_img = img_matrix[y_min:y_max, x_min:x_max]
            
            # Mã hóa ảnh thẳng trên RAM thành chuẩn JPEG
            success, buffer = cv2.imencode(".jpg", crop_img)
            minio_url = None
            if success:
                image_bytes = buffer.tobytes()
                filename = f"page{page_num}_index{index}_{lbl}.jpg"
                object_path = self.minio_service.build_object_path(user_id, conv_id, doc_id, filename)
                
                # Upload thẳng stream lên MinIO
                minio_url = self.minio_service.upload_image_bytes(object_path, image_bytes)

            final_item = {
                "page": page_num,
                "index": index,
                **element,             # Bên trong element này đã chứa sẵn content (nếu có)
                "file_path": minio_url # Luôn luôn trả về đường dẫn ảnh cho Frontend
            }
            final_elements.append(final_item)
            
        return final_elements