# Build mineru-cli image
```bash
docker build --no-cache -t pp-ocr:v1 .
```
# Run
```bash
docker run --rm --name pp_ocr_service --gpus '"device=1"' --shm-size=4gb --ipc=host -p 8201:8201 -v "$(pwd)/server.py":/app/server.py -e OCR_NUM_WORKERS=1 -e OCR_VERSION=PP-OCRv4 -e OCR_LANG=vi -e CUDA_VISIBLE_DEVICES=0 pp-ocr:v1

docker run --rm --name pp_ocr_service --gpus '"device=1"' --shm-size=4gb --ipc=host -p 8201:8201 -v "$(pwd)/server.py":/app/server.py -e OCR_NUM_WORKERS=1 -e OCR_VERSION=PP-OCRv4 -e OCR_LANG=vi -e CUDA_VISIBLE_DEVICES=1 -e FLAGS_allocator_strategy=auto_growth --restart always pp-ocr:v1

docker rm -f pp_ocr_service

docker run --rm --name pp_ocr_service --gpus '"device=1"' --shm-size=16gb --ipc=host -p 8201:8201 -v "$(pwd)/server.py":/app/server.py -e OCR_NUM_WORKERS=4 -e OCR_USE_ANGLE_CLS=true -e CUDA_VISIBLE_DEVICES=0 -e FLAGS_allocator_strategy=auto_growth pp-ocr:v1
```

# Curl pp-ocr
```bash
curl -X POST "http://localhost:8201/api/ocr/pp-ocr" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@anh_crop_thu_nghiem.jpg"
```
