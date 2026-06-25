import base64
import requests
import os

class LlmOcrService:
    def __init__(self):
        self.api_url = os.getenv("LLM_OCR_ENDPOINT", "http://localhost:8300/v1/chat/completions")
        self.model = os.getenv("LLM_OCR_MODEL", "paddleocr")
        self.temperature = float(os.getenv("LLM_OCR_TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("LLM_OCR_MAX_TOKENS", "4096"))

    def _get_base64_from_local(self, file_path):
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def request_ocr(self, image_input, is_local=True, is_base64=False):
        if is_base64:
            image_url = f"data:image/jpeg;base64,{image_input}"
        elif is_local:
            image_url = f"data:image/jpeg;base64,{self._get_base64_from_local(image_input)}"
        else:
            image_url = image_input

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ]
                }
            ]
        }

        max_retries = 3
        timeout_seconds = 60
        import time

        for attempt in range(max_retries):
            try:
                print(f"[LLM OCR] Sending request to VLM API (attempt {attempt + 1}/{max_retries})...", flush=True)
                response = requests.post(self.api_url, json=payload, timeout=timeout_seconds)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                if content:
                    return content
                print(f"[LLM OCR] Attempt {attempt + 1}/{max_retries}: VLM API returned empty content.", flush=True)
            except Exception as e:
                print(f"[LLM OCR] Attempt {attempt + 1}/{max_retries} failed: {e}", flush=True)
            
            if attempt < max_retries - 1:
                # Chờ tăng dần (exponential backoff) trước khi thử lại
                sleep_time = (attempt + 1) * 2
                print(f"[LLM OCR] Retrying in {sleep_time} seconds...", flush=True)
                time.sleep(sleep_time)

        print("[LLM OCR] All retry attempts failed or VLM returned empty content.", flush=True)
        return ""