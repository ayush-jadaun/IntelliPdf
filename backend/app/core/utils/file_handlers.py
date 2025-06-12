"""
Enhanced File Handling Utilities for IntelliPDF
Improved version with better error handling, security, async support, validation, and MinIO integration.
"""

import os
import shutil
import uuid
import hashlib
import mimetypes
import logging
from pathlib import Path
from typing import Optional, Union, Callable, Dict, Any, List
from contextlib import contextmanager
import asyncio
import aiofiles
import re
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# Configure logging
logger = logging.getLogger(__name__)

# Configuration
class FileConfig:
    ALLOWED_EXTENSIONS = {".pdf"}
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    CHUNK_SIZE = 8192  # 8KB chunks for streaming
    ALLOWED_MIME_TYPES = {"application/pdf"}

class MinIOConfig:
    """MinIO configuration - should be imported from app.config"""
    ENDPOINT = None  # Set from app.config
    ACCESS_KEY = None  # Set from app.config
    SECRET_KEY = None  # Set from app.config
    BUCKET = None  # Set from app.config

# Custom Exceptions
class FileHandlingError(Exception):
    """Base exception for file handling operations."""
    pass

class InvalidFileTypeError(FileHandlingError):
    """Raised when file type is not allowed."""
    pass

class FileSizeExceededError(FileHandlingError):
    """Raised when file size exceeds limits."""
    pass

class FileValidationError(FileHandlingError):
    """Raised when file validation fails."""
    pass

class CloudStorageError(FileHandlingError):
    """Raised when cloud storage operations fail."""
    pass

def init_minio_config(endpoint: str, access_key: str, secret_key: str, bucket: str):
    """Initialize MinIO configuration."""
    MinIOConfig.ENDPOINT = endpoint
    MinIOConfig.ACCESS_KEY = access_key
    MinIOConfig.SECRET_KEY = secret_key
    MinIOConfig.BUCKET = bucket

def get_s3_client():
    """Get configured S3 client for MinIO."""
    if not all([MinIOConfig.ENDPOINT, MinIOConfig.ACCESS_KEY, MinIOConfig.SECRET_KEY]):
        raise CloudStorageError("MinIO configuration not initialized")
    
    try:
        return boto3.client(
            "s3",
            endpoint_url=f"http://{MinIOConfig.ENDPOINT}",
            aws_access_key_id=MinIOConfig.ACCESS_KEY,
            aws_secret_access_key=MinIOConfig.SECRET_KEY,
        )
    except Exception as e:
        raise CloudStorageError(f"Failed to create S3 client: {e}")

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and other security issues."""
    # Remove path components
    filename = os.path.basename(filename)
    
    # Remove dangerous characters
    dangerous_chars = r'[<>:"/\\|?*\x00-\x1f]'
    filename = re.sub(dangerous_chars, '_', filename)
    
    # Limit length
    name, ext = os.path.splitext(filename)
    if len(name) > 100:
        name = name[:100]
    
    return name + ext

def is_allowed_file(filename: str) -> bool:
    """Check if the file has an allowed extension and MIME type."""
    file_path = Path(filename)
    extension = file_path.suffix.lower()
    
    if extension not in FileConfig.ALLOWED_EXTENSIONS:
        return False
    
    # Check MIME type
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type not in FileConfig.ALLOWED_MIME_TYPES:
        return False
    
    return True

def validate_pdf_content(file_data: Union[str, bytes]) -> bool:
    """Validate that the file is actually a PDF by checking its header."""
    try:
        if isinstance(file_data, str):
            # File path
            with open(file_data, 'rb') as f:
                header = f.read(4)
        else:
            # Bytes data
            header = file_data[:4] if len(file_data) >= 4 else b''
        
        return header == b'%PDF'
    except Exception as e:
        logger.error(f"Error validating PDF content: {e}")
        return False

def get_file_hash(file_data: Union[str, bytes], algorithm: str = 'sha256') -> str:
    """Generate hash of file content for integrity checking."""
    hash_obj = hashlib.new(algorithm)
    
    if isinstance(file_data, str):
        # File path
        with open(file_data, 'rb') as f:
            for chunk in iter(lambda: f.read(FileConfig.CHUNK_SIZE), b''):
                hash_obj.update(chunk)
    else:
        # Bytes data
        hash_obj.update(file_data)
    
    return hash_obj.hexdigest()

def get_unique_filename(filename: str, use_timestamp: bool = False) -> str:
    """Generate a unique filename with better collision avoidance."""
    sanitized = sanitize_filename(filename)
    file_path = Path(sanitized)
    
    if use_timestamp:
        import time
        unique_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    else:
        unique_id = uuid.uuid4().hex[:12]
    
    return f"{file_path.stem}_{unique_id}{file_path.suffix}"

def validate_file_size(file_data: Union[bytes, Any], max_size: int = None) -> bool:
    """Validate file size before processing."""
    max_size = max_size or FileConfig.MAX_FILE_SIZE
    
    if isinstance(file_data, bytes):
        return len(file_data) <= max_size
    elif hasattr(file_data, 'seek') and hasattr(file_data, 'tell'):
        # For file-like objects
        current_pos = file_data.tell()
        file_data.seek(0, 2)  # Seek to end
        size = file_data.tell()
        file_data.seek(current_pos)  # Restore position
        return size <= max_size
    
    return True  # Can't determine size, allow

def upload_to_minio(file_bytes: bytes, filename: str, metadata: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Upload file to MinIO with validation and metadata.
    
    Args:
        file_bytes: File content as bytes
        filename: Original filename
        metadata: Optional metadata dictionary
    
    Returns:
        Dictionary with upload info
    """
    try:
        # Validate file
        if not is_allowed_file(filename):
            raise InvalidFileTypeError(f"File type not allowed: {filename}")
        
        if not validate_file_size(file_bytes):
            raise FileSizeExceededError("File size exceeds maximum allowed size")
        
        if not validate_pdf_content(file_bytes):
            raise FileValidationError("File content validation failed - not a valid PDF")
        
        # Generate unique filename for storage
        unique_filename = get_unique_filename(filename)
        
        # Prepare metadata
        file_metadata = {
            'original-name': filename,
            'content-type': 'application/pdf',
            'upload-timestamp': str(int(asyncio.get_event_loop().time() * 1000)),
            'file-hash': get_file_hash(file_bytes)
        }
        
        if metadata:
            file_metadata.update(metadata)
        
        # Upload to MinIO
        s3_client = get_s3_client()
        s3_client.put_object(
            Bucket=MinIOConfig.BUCKET,
            Key=unique_filename,
            Body=file_bytes,
            ContentType='application/pdf',
            Metadata=file_metadata
        )
        
        upload_info = {
            'original_name': filename,
            'stored_name': unique_filename,
            'size': len(file_bytes),
            'hash': file_metadata['file-hash'],
            'bucket': MinIOConfig.BUCKET,
            'metadata': file_metadata
        }
        
        logger.info(f"File uploaded to MinIO successfully: {unique_filename}")
        return upload_info
        
    except ClientError as e:
        logger.error(f"MinIO client error uploading {filename}: {e}")
        raise CloudStorageError(f"Failed to upload to cloud storage: {e}")
    except Exception as e:
        logger.error(f"Error uploading file {filename} to MinIO: {e}")
        raise

def download_from_minio(filename: str, validate_integrity: bool = True) -> Dict[str, Any]:
    """
    Download file from MinIO with optional integrity validation.
    
    Args:
        filename: Stored filename in MinIO
        validate_integrity: Whether to validate file integrity using stored hash
    
    Returns:
        Dictionary with file data and metadata
    """
    try:
        s3_client = get_s3_client()
        
        # Get object and metadata
        response = s3_client.get_object(Bucket=MinIOConfig.BUCKET, Key=filename)
        file_bytes = response["Body"].read()
        
        # Extract metadata
        metadata = response.get('Metadata', {})
        
        # Validate integrity if requested
        if validate_integrity and 'file-hash' in metadata:
            current_hash = get_file_hash(file_bytes)
            stored_hash = metadata['file-hash']
            
            if current_hash != stored_hash:
                raise FileValidationError(f"File integrity check failed for {filename}")
        
        download_info = {
            'filename': filename,
            'original_name': metadata.get('original-name', filename),
            'data': file_bytes,
            'size': len(file_bytes),
            'metadata': metadata,
            'content_type': response.get('ContentType', 'application/pdf')
        }
        
        logger.info(f"File downloaded from MinIO successfully: {filename}")
        return download_info
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            raise FileHandlingError(f"File not found in cloud storage: {filename}")
        logger.error(f"MinIO client error downloading {filename}: {e}")
        raise CloudStorageError(f"Failed to download from cloud storage: {e}")
    except Exception as e:
        logger.error(f"Error downloading file {filename} from MinIO: {e}")
        raise

def delete_from_minio(filename: str) -> bool:
    """Delete file from MinIO."""
    try:
        s3_client = get_s3_client()
        s3_client.delete_object(Bucket=MinIOConfig.BUCKET, Key=filename)
        
        logger.info(f"File deleted from MinIO: {filename}")
        return True
        
    except ClientError as e:
        logger.error(f"MinIO client error deleting {filename}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error deleting file {filename} from MinIO: {e}")
        return False

def list_files_in_minio(prefix: str = "", max_keys: int = 1000) -> List[Dict[str, Any]]:
    """List files in MinIO bucket with optional prefix filter."""
    try:
        s3_client = get_s3_client()
        
        kwargs = {
            'Bucket': MinIOConfig.BUCKET,
            'MaxKeys': max_keys
        }
        
        if prefix:
            kwargs['Prefix'] = prefix
        
        response = s3_client.list_objects_v2(**kwargs)
        
        files = []
        for obj in response.get('Contents', []):
            # Get object metadata
            try:
                head_response = s3_client.head_object(Bucket=MinIOConfig.BUCKET, Key=obj['Key'])
                metadata = head_response.get('Metadata', {})
            except:
                metadata = {}
            
            files.append({
                'key': obj['Key'],
                'original_name': metadata.get('original-name', obj['Key']),
                'size': obj['Size'],
                'last_modified': obj['LastModified'],
                'metadata': metadata
            })
        
        return files
        
    except ClientError as e:
        logger.error(f"MinIO client error listing files: {e}")
        raise CloudStorageError(f"Failed to list files from cloud storage: {e}")
    except Exception as e:
        logger.error(f"Error listing files in MinIO: {e}")
        raise

def save_uploaded_file(
    upload_dir: str,
    file_data: Union[bytes, Any],
    filename: str,
    validate_content: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    save_to_cloud: bool = False
) -> Dict[str, Any]:
    """
    Enhanced file saving with validation, progress tracking, metadata, and optional cloud storage.
    
    Args:
        upload_dir: Local directory to save file
        file_data: File data (bytes or file-like object)
        filename: Original filename
        validate_content: Whether to validate PDF content
        progress_callback: Progress callback function
        save_to_cloud: Whether to also save to MinIO cloud storage
    
    Returns:
        Dictionary with file info including cloud storage info if applicable
    """
    try:
        # Validate inputs
        if not is_allowed_file(filename):
            raise InvalidFileTypeError(f"File type not allowed: {filename}")
        
        if not validate_file_size(file_data):
            raise FileSizeExceededError("File size exceeds maximum allowed size")
        
        # Convert file data to bytes for processing
        if hasattr(file_data, "read"):
            file_bytes = file_data.read()
            if hasattr(file_data, 'seek'):
                file_data.seek(0)  # Reset for potential reuse
        elif isinstance(file_data, bytes):
            file_bytes = file_data
        else:
            raise ValueError("Unsupported file_data type")
        
        # Validate content if requested
        if validate_content and not validate_pdf_content(file_bytes):
            raise FileValidationError("File content validation failed")
        
        # Ensure directory exists
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        unique_filename = get_unique_filename(filename)
        save_path = os.path.join(upload_dir, unique_filename)
        
        # Save file locally with progress tracking
        total_size = len(file_bytes)
        
        with open(save_path, "wb") as out_file:
            for i in range(0, total_size, FileConfig.CHUNK_SIZE):
                chunk = file_bytes[i:i + FileConfig.CHUNK_SIZE]
                out_file.write(chunk)
                
                if progress_callback:
                    progress_callback(i + len(chunk), total_size)
        
        # Generate file metadata
        file_info = {
            'original_name': filename,
            'saved_name': unique_filename,
            'local_path': save_path,
            'size': total_size,
            'hash': get_file_hash(file_bytes),
            'mime_type': mimetypes.guess_type(filename)[0]
        }
        
        # Save to cloud storage if requested
        if save_to_cloud:
            try:
                cloud_info = upload_to_minio(file_bytes, filename)
                file_info['cloud_storage'] = cloud_info
                file_info['cloud_key'] = cloud_info['stored_name']
            except Exception as e:
                logger.warning(f"Failed to save to cloud storage: {e}")
                file_info['cloud_storage_error'] = str(e)
        
        logger.info(f"File saved successfully: {unique_filename}")
        return file_info
        
    except Exception as e:
        logger.error(f"Error saving file {filename}: {e}")
        # Clean up partial file if it exists
        if 'save_path' in locals() and os.path.exists(save_path):
            os.remove(save_path)
        raise

async def save_uploaded_file_async(
    upload_dir: str,
    file_data: Union[bytes, Any],
    filename: str,
    validate_content: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    save_to_cloud: bool = False
) -> Dict[str, Any]:
    """Async version of save_uploaded_file."""
    try:
        # Validate inputs (same as sync version)
        if not is_allowed_file(filename):
            raise InvalidFileTypeError(f"File type not allowed: {filename}")
        
        if not validate_file_size(file_data):
            raise FileSizeExceededError("File size exceeds maximum allowed size")
        
        # Convert to bytes
        if isinstance(file_data, bytes):
            file_bytes = file_data
        else:
            # For file-like objects, fall back to sync version
            return save_uploaded_file(upload_dir, file_data, filename, validate_content, progress_callback, save_to_cloud)
        
        # Validate content if requested
        if validate_content and not validate_pdf_content(file_bytes):
            raise FileValidationError("File content validation failed")
        
        # Ensure directory exists
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        unique_filename = get_unique_filename(filename)
        save_path = os.path.join(upload_dir, unique_filename)
        
        # Save file asynchronously
        async with aiofiles.open(save_path, "wb") as out_file:
            await out_file.write(file_bytes)
        
        # Generate file metadata
        file_info = {
            'original_name': filename,
            'saved_name': unique_filename,
            'local_path': save_path,
            'size': len(file_bytes),
            'hash': get_file_hash(file_bytes),
            'mime_type': mimetypes.guess_type(filename)[0]
        }
        
        # Save to cloud storage if requested
        if save_to_cloud:
            try:
                cloud_info = upload_to_minio(file_bytes, filename)
                file_info['cloud_storage'] = cloud_info
                file_info['cloud_key'] = cloud_info['stored_name']
            except Exception as e:
                logger.warning(f"Failed to save to cloud storage: {e}")
                file_info['cloud_storage_error'] = str(e)
        
        logger.info(f"File saved successfully (async): {unique_filename}")
        return file_info
        
    except Exception as e:
        logger.error(f"Error saving file async {filename}: {e}")
        if 'save_path' in locals() and os.path.exists(save_path):
            os.remove(save_path)
        raise

def delete_file(file_path: str, secure: bool = False, delete_from_cloud: bool = False, cloud_key: str = None) -> bool:
    """
    Delete a file with optional secure deletion and cloud storage cleanup.
    
    Args:
        file_path: Path to local file to delete
        secure: If True, overwrite file before deletion
        delete_from_cloud: Whether to also delete from cloud storage
        cloud_key: Key of file in cloud storage (if different from filename)
    """
    success = True
    
    try:
        # Delete local file
        if os.path.exists(file_path):
            if secure:
                # Overwrite file with random data before deletion
                file_size = os.path.getsize(file_path)
                with open(file_path, 'wb') as f:
                    f.write(os.urandom(file_size))
            
            os.remove(file_path)
            logger.info(f"Local file deleted: {file_path}")
        
        # Delete from cloud storage if requested
        if delete_from_cloud:
            key_to_delete = cloud_key or os.path.basename(file_path)
            cloud_success = delete_from_minio(key_to_delete)
            success = success and cloud_success
        
        return success
        
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        return False

def get_file_info(file_path: str) -> Optional[Dict[str, Any]]:
    """Get comprehensive file information."""
    try:
        if not os.path.exists(file_path):
            return None
        
        stat = os.stat(file_path)
        
        return {
            'path': file_path,
            'name': os.path.basename(file_path),
            'size': stat.st_size,
            'created': stat.st_ctime,
            'modified': stat.st_mtime,
            'accessed': stat.st_atime,
            'mime_type': mimetypes.guess_type(file_path)[0],
            'hash': get_file_hash(file_path)
        }
        
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {e}")
        return None

def move_file_safe(src_path: str, dest_dir: str, overwrite: bool = False) -> str:
    """Move file with collision handling and verification."""
    try:
        os.makedirs(dest_dir, exist_ok=True)
        
        dest_filename = os.path.basename(src_path)
        dest_path = os.path.join(dest_dir, dest_filename)
        
        # Handle filename collision
        if os.path.exists(dest_path) and not overwrite:
            base, ext = os.path.splitext(dest_filename)
            counter = 1
            while os.path.exists(dest_path):
                dest_filename = f"{base}_{counter}{ext}"
                dest_path = os.path.join(dest_dir, dest_filename)
                counter += 1
        
        # Get source hash for verification
        src_hash = get_file_hash(src_path)
        
        # Move file
        shutil.move(src_path, dest_path)
        
        # Verify integrity
        dest_hash = get_file_hash(dest_path)
        if src_hash != dest_hash:
            raise FileHandlingError("File integrity verification failed after move")
        
        logger.info(f"File moved successfully: {src_path} -> {dest_path}")
        return dest_path
        
    except Exception as e:
        logger.error(f"Error moving file {src_path} to {dest_dir}: {e}")
        raise

@contextmanager
def temp_file_cleanup(*file_paths):
    """Context manager for automatic cleanup of temporary files."""
    try:
        yield
    finally:
        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"Cleaned up temp file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to clean up temp file {file_path}: {e}")

def ensure_directory(dir_path: str, permissions: int = 0o755) -> None:
    """Ensure directory exists with proper permissions."""
    try:
        os.makedirs(dir_path, mode=permissions, exist_ok=True)
        logger.debug(f"Directory ensured: {dir_path}")
    except Exception as e:
        logger.error(f"Error ensuring directory {dir_path}: {e}")
        raise

# Convenience function for easy integration
def setup_file_handler(minio_endpoint: str, minio_access_key: str, minio_secret_key: str, minio_bucket: str):
    """
    Setup file handler with MinIO configuration.
    Call this function at application startup.
    """
    init_minio_config(minio_endpoint, minio_access_key, minio_secret_key, minio_bucket)
    logger.info("File handler initialized with MinIO support")