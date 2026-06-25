import os
import json
import io
from minio import Minio
from minio.error import S3Error

class MinioService:
    def __init__(self):
        self.endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "document_storage")
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "document_storage_password_123")
        self.secure = os.getenv("MINIO_SECURE", "False").lower() == "true"
        self.bucket_name = "documents-ocr"

        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
        except S3Error as err:
            print(f"MinIO Bucket Error: {err}")

    def build_object_path(self, user_id, conversation_id, document_id, filename):
        return f"{user_id}/{conversation_id}/{document_id}/{filename}"

    def upload_image_bytes(self, object_name, image_bytes):
        try:
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=io.BytesIO(image_bytes),
                length=len(image_bytes),
                content_type="image/jpeg"
            )
            return f"{self.bucket_name}/{object_name}"
        except S3Error as err:
            print(f"MinIO Upload Error: {err}")
            return None

    def upload_json(self, object_name, json_data):
        try:
            json_bytes = json.dumps(json_data, ensure_ascii=False, indent=4).encode('utf-8')
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=io.BytesIO(json_bytes),
                length=len(json_bytes),
                 content_type="application/json"
            )
            return f"{self.bucket_name}/{object_name}"
        except S3Error as err:
            print(f"MinIO Upload JSON Error: {err}")
            return None

    def upload_file_bytes(self, object_name, file_bytes, content_type="application/octet-stream"):
        try:
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=io.BytesIO(file_bytes),
                length=len(file_bytes),
                content_type=content_type
            )
            return f"{self.bucket_name}/{object_name}"
        except S3Error as err:
            print(f"MinIO Upload File Error: {err}")
            return None