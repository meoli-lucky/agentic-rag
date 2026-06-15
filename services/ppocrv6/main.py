import os
import cv2
from paddleocr import PaddleOCR

def main():
    print("Đang khởi tạo toàn bộ Pipeline PP-OCRv6 (Det + Rec) trên GPU...")
    
    ocr = PaddleOCR(
                                             
        text_detection_model_name="PP-OCRv6_medium_det",    
        text_recognition_model_name="PP-OCRv6_medium_rec",  
        engine="transformers",                              
        device="gpu:0",                                     
        use_doc_orientation_classify=False,                 
        use_doc_unwarping=False,                            
        use_textline_orientation=False                      
    )

    folder_path = "./test_images"
    if not os.path.exists(folder_path):
        print(f"Không tìm thấy thư mục {folder_path}.")
        return

    image_paths = [
        os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path))
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ]

    if not image_paths:
        print("Không có ảnh hợp lệ nào trong thư mục test.")
        return

    print(f"Đã nạp {len(image_paths)} ảnh. Bắt đầu nhận diện và ghi file...")

    # Tạo đường dẫn lưu file kết quả nằm ngay trong thư mục mount để dễ lấy
    output_txt_path = os.path.join(folder_path, "ket_qua_ocr.txt")

    # Mở file ở chế độ ghi ('w') với chuẩn utf-8 cho tiếng Việt
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("--- KẾT QUẢ OCR ---\n\n")
        print("\n--- KẾT QUẢ OCR ---")
        
        for img_path in image_paths:
            file_name = os.path.basename(img_path)
            
            # Vừa in ra console để theo dõi, vừa ghi vào file
            print(f"\n[{file_name}]:")
            f.write(f"[{file_name}]:\n")
            
            try:
                results = ocr.predict(img_path)
                
                for res in results:
                    try:
                        res_dict = dict(res)
                    except Exception:
                        res_dict = res
                    
                    if 'rec_texts' in res_dict:
                        texts = res_dict['rec_texts']
                        scores = res_dict['rec_scores']
                    else:
                        error_msg = f"  + Lỗi định dạng từ mô hình. Cấu trúc gốc: {res_dict.keys() if isinstance(res_dict, dict) else res}"
                        print(error_msg)
                        f.write(error_msg + "\n")
                        continue
                        
                    if not texts or len(texts) == 0:
                        empty_msg = "  + Không tìm thấy đoạn text nào trong ảnh này."
                        print(empty_msg)
                        f.write(empty_msg + "\n")
                        continue
                    
                    # Ghi từng dòng văn bản
                    for idx, (van_ban, do_tin_cay) in enumerate(zip(texts, scores)):
                        line_result = f"  + Dòng {idx+1}: {van_ban} (Độ tự tin: {float(do_tin_cay):.2f})"
                        print(line_result)
                        f.write(line_result + "\n")
                        
            except Exception as e:
                err_msg = f"  + KHÔNG ĐỌC ĐƯỢC - Lỗi: {e}"
                print(err_msg)
                f.write(err_msg + "\n")
            
            # Xuống dòng cách điệu giữa các ảnh cho dễ đọc
            f.write("\n")

    print(f"\nĐã nhận diện xong! Toàn bộ kết quả được lưu tại: {output_txt_path}")

if __name__ == "__main__":
    main()