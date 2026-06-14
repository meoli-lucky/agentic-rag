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

        # Cấu trúc body linh hoạt
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

        try:
            # Phương thức POST mặc định
            response = requests.post(self.api_url, json=payload, timeout=45)
            response.raise_for_status() # Kiểm tra lỗi HTTP
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"LLM OCR Error: {e}")
            return None