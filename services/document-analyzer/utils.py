def get_extended_bbox(bbox, img_height, img_width, extend_border):
    """
    Hàm tính toán tọa độ bbox mới sau khi mở rộng biên (extend border).
    extend_border: định dạng "top,left,bottom,right" (ví dụ: "5,5,5,5" hoặc "0.05,0.02,0.05,0.02")
    """
    if not extend_border:
        return bbox
    try:
        if isinstance(extend_border, str):
            # Tách chuỗi và loại bỏ khoảng trắng
            parts = [float(x.strip()) for x in extend_border.split(",")]
        else:
            parts = list(extend_border)
        
        if len(parts) != 4:
            return bbox
            
        t, l, b, r = parts
        x_min, y_min, x_max, y_max = bbox
        w = x_max - x_min
        h = y_max - y_min
        
        # Nhận diện đơn vị: < 1.0 là tỷ lệ %, >= 1.0 là pixel
        top_pad = int(t * h) if t < 1.0 else int(t)
        left_pad = int(l * w) if l < 1.0 else int(l)
        bottom_pad = int(b * h) if b < 1.0 else int(b)
        right_pad = int(r * w) if r < 1.0 else int(r)
        
        # Áp dụng mở rộng và giới hạn tọa độ trong kích thước ảnh để tránh lỗi index âm hoặc tràn biên
        new_x_min = max(0, x_min - left_pad)
        new_y_min = max(0, y_min - top_pad)
        new_x_max = min(img_width, x_max + right_pad)
        new_y_max = min(img_height, y_max + bottom_pad)
        
        return [new_x_min, new_y_min, new_x_max, new_y_max]
    except Exception as e:
        print(f"[Extend Border] Error parsing extend_border '{extend_border}': {e}")
        return bbox
