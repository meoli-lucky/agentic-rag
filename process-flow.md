# Document Analyzer — Process Flow

Tài liệu mô tả luồng xử lý và các điều kiện logic của service `document-analyzer`.

---

## 1. Tổng quan kiến trúc

```
┌─────────────────────────────────────────────────────────┐
│                       FastAPI (app.py)                  │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────┐  │
│  │  YOLOv10    │  │CoordinateProcessor│  │NativeText │  │
│  │ (YOLO Model)│  │                  │  │ Analyzer  │  │
│  └─────────────┘  └──────────────────┘  └───────────┘  │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              SmartOcrService                        │ │
│  │   ┌────────────────────────────────────────────┐   │ │
│  │   │           LlmOcrService                    │   │ │
│  │   │   (HTTP → llama-server / paddleocr-vl)     │   │ │
│  │   └────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────┘ │
│  ┌────────────────────┐  ┌────────────────────────────┐  │
│  │ ImageCropperS3     │  │ ImageCropperLocal           │  │
│  │ (MinioService)     │  │ (/app/output/<request_id>) │  │
│  └────────────────────┘  └────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 2. API Endpoint

**POST** `/api/v1/layout-analysis`

### 2.1 Headers (bắt buộc)

| Header               | Mô tả                     |
|----------------------|---------------------------|
| `x-user-id`          | UUID của User             |
| `x-conversation-id`  | ID phiên chat / hồ sơ    |
| `x-document-id`      | ID tài liệu               |

### 2.2 Form Parameters

| Tham số               | Kiểu    | Mặc định       | Mô tả                                                                 |
|-----------------------|---------|----------------|-----------------------------------------------------------------------|
| `file`                | File    | —              | File PDF hoặc ảnh đầu vào                                             |
| `storage_type`        | string  | `"s3"`         | Đích lưu ảnh crop và JSON: `"s3"` hoặc `"local"`                    |
| `confidence_threshold`| float   | `0.25`         | Ngưỡng confidence tối thiểu cho YOLO                                  |
| `sort`                | string  | `"coordinates"`| Cách sắp xếp kết quả: `"coordinates"` hoặc `"confidence"`            |
| `show_height`         | bool    | `false`        | Có trả về `box_height` trong JSON không                               |
| `show_width`          | bool    | `false`        | Có trả về `box_width` trong JSON không                                |
| `remove_page_header`  | bool    | `false`        | Bỏ qua các box có label `Page-header` không                           |
| `merge_suspicion`     | bool    | `true`         | Ghi lại các box bị loại (IoU overlap) vào field `suspicion`           |
| `check_digital_text`  | bool    | `true`         | Trích xuất text gốc từ PDF (native text extraction)                   |
| `doc_recognizer`      | bool    | `true`         | Kích hoạt nhận dạng nội dung (text / OCR). Nếu `true` → ghi đè `check_digital_text=true` |
| `table_recognizer`    | bool    | `true`         | OCR các vùng Table. Nếu `false` → `content=""` với box Table          |
| `smart_ocr`           | bool    | `true`         | Dùng Smart OCR (ghép ảnh, OCR 1 lần / page) hay OCR từng ảnh nhỏ    |

---

## 3. Luồng xử lý chính

### Bước 0 — Tiền xử lý tham số

```
doc_recognizer == true ?
    ├── YES → check_digital_text = true  (ghi đè)
    └── NO  → giữ nguyên check_digital_text
```

> **Lý do**: Nếu `doc_recognizer=true` mà `check_digital_text=false` thì không xác định được vùng nào cần OCR — buộc phải bật digital text detection.

---

### Bước 1 — Đọc & render PDF → ảnh

Dùng **PyMuPDF (fitz)** để render từng trang PDF thành ảnh:

```python
pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))   # scale ×3
```

- Scale factor `3×` để tăng độ phân giải cho YOLO detect chính xác hơn.
- Tọa độ bbox YOLO sẽ ở hệ tọa độ ảnh ×3 → cần chia `scale_factor=3.0` để quy đổi về tọa độ PDF gốc (dùng cho native text extraction).

---

### Bước 2 — Layout Detection (YOLOv10)

```
YOLOv10(img_bgr, imgsz=1024, conf=confidence_threshold, iou=0.45, agnostic_nms=True)
```

Mỗi box detected gồm: `label`, `confidence`, `bbox [x_min, y_min, x_max, y_max]`, `box_width`, `box_height`.

**Các label YOLO hỗ trợ**: `Text`, `Section-header`, `List-item`, `Table`, `Page-header`, v.v.

**Log**: `[Page N] Layout detection took X.XXs (found N boxes)`

---

### Bước 3 — Xử lý tọa độ (CoordinateProcessor)

Pipeline 4 bước:

```
raw_elements
  │
  ├─ [1] remove_page_header == true → lọc bỏ box label="Page-header"
  │
  ├─ [2] calculate_metrics_and_tags()
  │       ├── Tính aspect_ratio, estimated_lines, line_height
  │       └── Gán predicted_tag:
  │               Section-header → h1 / h2 / h3  (theo line_height)
  │               List-item      → list-item
  │               Table          → table
  │               Page-header    → ignore_header
  │               (khác)         → paragraph
  │
  ├─ [3] merge_duplicates()
  │       ├── Sort theo confidence (cao → thấp)
  │       ├── Tính IoU giữa các box
  │       ├── IoU > 0.85 → box trùng lấp → loại bỏ
  │       └── merge_suspicion == true → lưu box bị loại vào field "suspicion" của box giữ lại
  │
  └─ [4] sort == "coordinates" → sort_spatial() (thuật toán XY-Cut)
              ├── Sort sơ bộ theo trục Y
              ├── Nhóm các box cùng "dòng" (|y_diff| <= 15px)
              └── Trong mỗi dòng, sort theo trục X (trái → phải)
```

---

### Bước 4 — Native Text Extraction (NativeTextAnalyzer)

**Điều kiện kích hoạt**: `check_digital_text == true` (hoặc `doc_recognizer == true` ghi đè)

```
check_digital_text == true ?
    ├── YES →  Với mỗi element:
    │           ├── Quy đổi bbox về tọa độ PDF gốc (÷ 3.0)
    │           ├── page.get_textbox(pdf_rect) → lấy text
    │           ├── text != "" → digital_text="true",  content=<text>
    │           └── text == "" → digital_text="false", content=""
    │
    └── NO  →  Tất cả element: digital_text="unverified", content=""
```

> `digital_text="false"` ≠ `content=""`. Nó có nghĩa là vùng đó **không có text gốc** và cần dùng OCR bên ngoài.

---

### Bước 5 — Chuẩn bị danh sách OCR

**Điều kiện kích hoạt**: `doc_recognizer == true`

```
Với mỗi element trong processed_elements:
    ├── label == "Table" AND table_recognizer == false
    │       → content = ""  (bỏ qua, không OCR)
    │
    └── digital_text == "false"
            → crop ảnh từ img_bgr[y_min:y_max, x_min:x_max]
            → thêm element vào ocr_elements[]
            → thêm crop_img vào ocr_crops[]
```

**Log**: `[Page N] JSON processing & crop preparation took X.XXs (prepared N crops for OCR)`

---

### Bước 6 — OCR (SmartOcrService)

**Điều kiện kích hoạt**: `doc_recognizer == true` AND `len(ocr_crops) > 0`

#### 6A. Smart OCR (`smart_ocr == true`) — **Luồng tối ưu**

```
is_pdf == true ?
    ├── chunk_size = len(ocr_crops)    # Toàn bộ crop của 1 page ghép thành 1 ảnh
    └── chunk_size = 10               # Mỗi 10 ảnh nhỏ ghép thành 1 ảnh (cho input ảnh)

Với mỗi chunk:
    1. Tìm max_width trong chunk
    2. Với mỗi crop trong chunk:
        ├── pad_image_to_width(crop, max_width)   # Đệm trắng bên phải
        └── thêm divider [$$$$$] giữa các ảnh
    3. np.vstack() → ảnh ghép (stacked_image)
    4. Lưu ảnh ghép:
        ├── storage_type == "s3"  → MinIO: <user>/<conv>/<doc>/page{N}_stacked.jpg
        └── storage_type == "local" → /app/output/<request_id>/page{N}_stacked.jpg
    5. OCR ảnh ghép 1 lần → ocr_text
    6. split_ocr_text(ocr_text, num_crops):
        ├── Tách theo pattern [$$$$$] (fuzzy regex)
        ├── Số đoạn < num_crops → bổ sung "" cho phần thiếu
        └── Số đoạn > num_crops → cắt bỏ phần thừa
    7. Gán content cho từng element tương ứng

Tên file ảnh ghép:
    ├── PDF  → page{N}_stacked.jpg
    └── Image → page{N}_stacked_part{M}.jpg
```

**Log**: `[Smart OCR] Chunk M (page N) took X.XXs, size: (H, W, C), crops: K`

#### 6B. Regular OCR (`smart_ocr == false`) — **Luồng cũ**

```
Với mỗi (element, crop_img):
    1. cv2.imencode(".jpg", crop_img) → bytes
    2. base64 encode
    3. LlmOcrService.request_ocr() → ocr_text
    4. element["content"] = ocr_text

Log: [Regular OCR] Crop K/N took X.XXs, size: (H, W, C)
```

---

### Bước 7 — Crop & Lưu ảnh nhỏ (Storage Routing)

Sau OCR, toàn bộ `processed_elements` (kể cả những box đã có `digital_text="true"`) được đưa qua cropper để lưu ảnh thumbnail:

```
storage_type == "s3" ?
    ├── YES → ImageCropperS3.crop_and_upload()
    │         → MinIO: <user_id>/<conv_id>/<doc_id>/page{N}_<label>_<idx>.jpg
    └── NO  → ImageCropperLocal.crop_and_save()
              → /app/output/<request_id>/page{N}_<label>_<idx>.jpg
```

---

### Bước 8 — Cleanup & Chuẩn hóa JSON output

```
Với mỗi item trong final_page_elements:
    ├── show_height == false → xóa field "box_height"
    ├── show_width  == false → xóa field "box_width"
    ├── xóa field "estimated_lines" (luôn)
    ├── doc_recognizer == false → content = ""  (ghi đè toàn bộ)
    └── content == None → content = ""
```

---

### Bước 9 — Lưu JSON kết quả

```json
{
  "status": "success",
  "document_metadata": {
    "user_id": "...",
    "conversation_id": "...",
    "document_id": "..."
  },
  "total_crops": <số element có file_path>,
  "result_file_url": "...",
  "data": [ <all_detected_data> ]
}
```

```
storage_type == "s3" ?
    ├── YES → MinIO: <user>/<conv>/<doc>/layout_analysis_result.json
    └── NO  → /app/output/<request_id>/layout_analysis_result.json
```

---

## 4. Sơ đồ luồng tổng hợp (Flow Diagram)

```
POST /api/v1/layout-analysis
         │
         ▼
[0] Normalize params
    doc_recognizer=true → check_digital_text=true
         │
         ▼
[1] Render PDF → img_bgr (scale ×3) [mỗi page]
         │
         ▼
[2] YOLOv10 Layout Detection
    → raw_elements [{label, confidence, bbox, ...}]
         │
         ▼
[3] CoordinateProcessor
    → filter header? → tag metrics → deduplicate (IoU) → spatial sort
    → processed_elements
         │
         ▼
[4] NativeTextAnalyzer
    check_digital_text=true?
    ├── YES → get_textbox() per element
    │         → digital_text="true"/"false", content=<text>/"">
    └── NO  → digital_text="unverified", content=""
         │
         ▼
[5] Chuẩn bị OCR list (nếu doc_recognizer=true)
    Với mỗi element có digital_text="false":
    ├── Table + table_recognizer=false → skip (content="")
    └── Còn lại → thêm vào [ocr_elements, ocr_crops]
         │
         ▼
[6] SmartOcrService.run_ocr() (nếu doc_recognizer=true)
    smart_ocr=true?
    ├── YES (Smart OCR):
    │   is_pdf=true?  chunk=all crops/page
    │   is_pdf=false? chunk=10 crops
    │   → stack images với divider [$$$$$]
    │   → lưu stacked image (s3/local)
    │   → OCR 1 lần / chunk → split → gán content
    └── NO (Regular OCR):
        → OCR từng crop riêng → gán content
         │
         ▼
[7] Crop & Store thumbnails (s3/local)
    ImageCropperS3 hoặc ImageCropperLocal
         │
         ▼
[8] Cleanup JSON fields
    (ẩn/xóa các field theo tham số)
         │
         ▼
[9] Lưu layout_analysis_result.json (s3/local)
         │
         ▼
    Trả về JSON response
```

---

## 5. Bảng điều kiện logic

| `doc_recognizer` | `check_digital_text` | `digital_text` (element) | Kết quả `content` |
|:---:|:---:|:---:|---|
| `false` | bất kỳ | `"unverified"` | `""` (ghi đè hoàn toàn) |
| `true` | `true` (auto) | `"true"` | Text PDF gốc |
| `true` | `true` (auto) | `"false"` | Kết quả từ OCR service |
| `true` | `false` | `"unverified"` | `""` |

| `smart_ocr` | Input | Cách tạo ảnh ghép |
|:---:|:---:|---|
| `true` | PDF | 1 ảnh ghép / page (tất cả crop của page đó) |
| `true` | Image | 1 ảnh ghép / 10 crops |
| `false` | bất kỳ | OCR từng crop riêng lẻ |

| `table_recognizer` | Element là Table | Kết quả |
|:---:|:---:|---|
| `true` | ✓ | OCR bình thường (nếu `digital_text="false"`) |
| `false` | ✓ | `content=""`, không OCR |

---

## 6. Cấu trúc thư mục lưu trữ

### S3 / MinIO

```
<user_id>/
  <conversation_id>/
    <document_id>/
      page1_Text_0.jpg
      page1_Table_1.jpg
      page1_stacked.jpg          ← Smart OCR stacked image (PDF)
      page2_Text_0.jpg
      page2_stacked.jpg
      layout_analysis_result.json
```

### Local Storage

```
/app/output/
  <request_id>/
    page1_Text_0.jpg
    page1_Table_1.jpg
    page1_stacked.jpg            ← Smart OCR stacked image (PDF)
    page1_stacked_part1.jpg      ← Smart OCR stacked image (Image input, chunk 1)
    layout_analysis_result.json
```

---

## 7. External Services

### LlmOcrService → llama-server (PaddleOCR-VL)

| Biến môi trường       | Mặc định                                     |
|-----------------------|----------------------------------------------|
| `LLM_OCR_ENDPOINT`    | `http://localhost:8300/v1/chat/completions`   |
| `LLM_OCR_MODEL`       | `paddleocr`                                   |
| `LLM_OCR_TEMPERATURE` | `0.1`                                         |
| `LLM_OCR_MAX_TOKENS`  | `4096`                                        |

> **Lưu ý Docker**: Khi chạy trong container, phải dùng `host.docker.internal` thay cho `localhost` để kết nối tới llama-server trên host.

### llama-server (khuyến nghị tham số)

```bash
llama-server \
  -m paddleocr-vl.gguf \
  --host 0.0.0.0 \
  --port 8300 \
  -fa on \       # Flash Attention
  -np 1 \        # 1 parallel slot (tránh thrash với Vision model)
  -c 8192        # Context size
```

> `-np 1` quan trọng: Chạy nhiều slot song song với Vision model dẫn đến thời gian xử lý tăng dần (từ 4s → >10s) do context thrashing.

---

## 8. Cấu trúc JSON output mỗi element

```json
{
  "label": "Text",
  "confidence": 0.92,
  "bbox": [120, 80, 900, 150],
  "box_width": 780,              // chỉ có nếu show_width=true
  "box_height": 70,              // chỉ có nếu show_height=true
  "predicted_tag": "paragraph",
  "line_height": 22.5,
  "digital_text": "true",        // "true" | "false" | "unverified"
  "content": "Nội dung văn bản...",
  "file_path": "s3://bucket/user/conv/doc/page1_Text_0.jpg",
  "suspicion": [                 // chỉ có nếu merge_suspicion=true
    {
      "label": "Section-header",
      "predicted_tag": "h2",
      "confidence": 0.61
    }
  ]
}
```
