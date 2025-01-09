import base64
import logging
import mimetypes
from io import BytesIO
import os
from pathlib import Path
import sys
from typing import List
from pdf2image import convert_from_path

sys.path.append(os.path.dirname('.'))

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Helper class for document processing operations"""
    
    @staticmethod
    def pdf_to_base64_images(file_path: str) -> List[str]:
        """Convert PDF pages to base64 encoded JPEG images."""
        try:
            images = convert_from_path(file_path)
            base64_images = []
            for image in images:
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                buffered.seek(0)
                base64_string = base64.b64encode(buffered.read()).decode("utf-8")
                base64_images.append(base64_string)
            return base64_images
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise

    @staticmethod
    def get_mime_type(file_path: str) -> str:
        """Determine MIME type of a file."""
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "application/octet-stream"

        
    # @staticmethod
    # def validate_mime_type(mime_type: str) -> bool:
    #     """Validate if MIME type is supported."""
    #     supported_types = ["image/jpeg", "image/png", "image/gif", "image/webp", "application/pdf"]
    #     return mime_type in supported_types