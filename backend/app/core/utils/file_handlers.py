import boto3
from app.config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET

s3 = boto3.client(
    "s3",
    endpoint_url=f"http://{MINIO_ENDPOINT}",
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
)

def upload_pdf(file_bytes: bytes, filename: str):
    s3.put_object(Bucket=MINIO_BUCKET, Key=filename, Body=file_bytes)

def download_pdf(filename: str):
    response = s3.get_object(Bucket=MINIO_BUCKET, Key=filename)
    return response["Body"].read()
