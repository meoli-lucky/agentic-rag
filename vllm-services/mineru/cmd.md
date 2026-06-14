# Build mineru-cli image
```bash
docker build -f Dockerfile-Cli --no-cache -t mineru-cli:v1 .
```

# Run test mineru-cli image
```bash
docker run --rm --gpus '"device=1"'  --shm-size=2gb -v "$(pwd)":/workspace -v ./magic-pdf.json:/root/magic-pdf.json -w /workspace   mineru-cli:v1 mineru -p 742.pdf -o .
docker run --rm --gpus '"device=1"'  --shm-size=2gb -v "$(pwd)":/workspace -v ./magic-pdf.json:/root/magic-pdf.json -w /workspace   mineru-cli:v1 mineru -p 1.jpeg -o .
```

# Build mineru-api-service image

```bash
docker build --no-cache -t mineru-api-service:v1 .
```
# Run mineru-api-service image
```bash
docker run -d --name mineru-api-container --gpus '"device=1"' --shm-size=2gb -p 8100:8100 -v "$(pwd)":/workspace -v ./magic-pdf.json:/root/magic-pdf.json -e MINERU_MODEL_SOURCE=local -e VLLM_GPU_MEMORY_UTILIZATION=0.5 -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 --ipc=host --restart always mineru-api-service:v1
```

# Curl mineru-api-service
```bash
curl --location --request POST 'http://103.147.123.161:8100/file_parse' \
--form 'files=@"C:\\Users\\meolilucky\\Downloads\\742.pdf"' \
--form 'lang_list="latin"' \
--form 'backend="pipeline"' \
--form 'parse_method="auto"' \
--form 'formula_enable="true"' \
--form 'table_enable="true"' \
--form 'image_analysis="true"' \
--form 'return_md="true"' \
--form 'return_middle_json="true"' \
--form 'return_model_output="true"' \
--form 'return_images="true"' \
--form 'response_format_zip="true"' \
--form 'return_original_file="true"' \
--form 'client_side_output_generation="false"'
```
