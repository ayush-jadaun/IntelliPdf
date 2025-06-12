"""
Enhanced File Handling Utilities for IntelliPDF
Improved version with better error handling, security, async support, and validation.
"""

import os
import shutil
import uuid
import hashlib
import mimetypes
import logging
from pathlib import Path
from typing import Optional, Union, Callable, Dict, Any
from contextlib import contextmanager
import asyncio
import aiofiles
import re
# Configure logging
logger = logging.getLogger(__name__)

# Configuration
class FileConfig:
    ALLOWED_EXTENSIONS = {".pdf"}
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    CHUNK_SIZE = 8192  # 8KB chunks for streaming
    ALLOWED_MIME_TYPES = {"application/pdf"}

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

def validate_pdf_content(file_path: str) -> bool:
    """Validate that the file is actually a PDF by checking its header."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF'
    except Exception as e:
        logger.error(f"Error validating PDF content: {e}")
        return False

def get_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
    """Generate hash of file content for integrity checking."""
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(FileConfig.CHUNK_SIZE), b''):
            hash_obj.update(chunk)
    
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

def save_uploaded_file(
    upload_dir: str,
    file_data: Union[bytes, Any],
    filename: str,
    validate_content: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, Any]:
    """
    Enhanced file saving with validation, progress tracking, and metadata.
    Returns dictionary with file info.
    """
    try:
        # Validate inputs
        if not is_allowed_file(filename):
            raise InvalidFileTypeError(f"File type not allowed: {filename}")
        
        if not validate_file_size(file_data):
            raise FileSizeExceededError("File size exceeds maximum allowed size")
        
        # Ensure directory exists
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        unique_filename = get_unique_filename(filename)
        save_path = os.path.join(upload_dir, unique_filename)
        
        # Save file with progress tracking
        total_size = 0
        
        if hasattr(file_data, "read"):
            # File-like object
            with open(save_path, "wb") as out_file:
                while True:
                    chunk = file_data.read(FileConfig.CHUNK_SIZE)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    total_size += len(chunk)
                    
                    if progress_callback:
                        progress_callback(total_size, total_size)
        
        elif isinstance(file_data, bytes):
            # Raw bytes
            total_size = len(file_data)
            with open(save_path, "wb") as out_file:
                for i in range(0, total_size, FileConfig.CHUNK_SIZE):
                    chunk = file_data[i:i + FileConfig.CHUNK_SIZE]
                    out_file.write(chunk)
                    
                    if progress_callback:
                        progress_callback(i + len(chunk), total_size)
        else:
            raise ValueError("Unsupported file_data type for saving")
        
        # Validate content if requested
        if validate_content and not validate_pdf_content(save_path):
            os.remove(save_path)
            raise FileValidationError("File content validation failed")
        
        # Generate file metadata
        file_info = {
            'original_name': filename,
            'saved_name': unique_filename,
            'path': save_path,
            'size': total_size,
            'hash': get_file_hash(save_path),
            'mime_type': mimetypes.guess_type(filename)[0]
        }
        
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
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, Any]:
    """Async version of save_uploaded_file."""
    try:
        # Validate inputs (same as sync version)
        if not is_allowed_file(filename):
            raise InvalidFileTypeError(f"File type not allowed: {filename}")
        
        if not validate_file_size(file_data):
            raise FileSizeExceededError("File size exceeds maximum allowed size")
        
        # Ensure directory exists
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        unique_filename = get_unique_filename(filename)
        save_path = os.path.join(upload_dir, unique_filename)
        
        # Save file asynchronously
        total_size = 0
        
        if isinstance(file_data, bytes):
            async with aiofiles.open(save_path, "wb") as out_file:
                await out_file.write(file_data)
                total_size = len(file_data)
        else:
            # For file-like objects, fall back to sync version
            return save_uploaded_file(upload_dir, file_data, filename, validate_content, progress_callback)
        
        # Validate content if requested
        if validate_content and not validate_pdf_content(save_path):
            os.remove(save_path)
            raise FileValidationError("File content validation failed")
        
        # Generate file metadata
        file_info = {
            'original_name': filename,
            'saved_name': unique_filename,
            'path': save_path,
            'size': total_size,
            'hash': get_file_hash(save_path),
            'mime_type': mimetypes.guess_type(filename)[0]
        }
        
        logger.info(f"File saved successfully (async): {unique_filename}")
        return file_info
        
    except Exception as e:
        logger.error(f"Error saving file async {filename}: {e}")
        if 'save_path' in locals() and os.path.exists(save_path):
            os.remove(save_path)
        raise

def delete_file(file_path: str, secure: bool = False) -> bool:
    """
    Delete a file with optional secure deletion.
    
    Args:
        file_path: Path to file to delete
        secure: If True, overwrite file before deletion
    """
    try:
        if not os.path.exists(file_path):
            return False
        
        if secure:
            # Overwrite file with random data before deletion
            file_size = os.path.getsize(file_path)
            with open(file_path, 'wb') as f:
                f.write(os.urandom(file_size))
        
        os.remove(file_path)
        logger.info(f"File deleted: {file_path}")
        return True
        
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