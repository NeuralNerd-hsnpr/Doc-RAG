"""
document_processor.py - Download and extract content from PDFs
Handles URL validation, PDF fetching, text extraction, and metadata management
"""

import requests
import pypdf
import logging
from typing import Optional, Dict, Tuple
from urllib.parse import urlparse
from datetime import datetime
import hashlib
import json
import os
import io
import traceback

from config import config

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Process documents from URLs:
    1. Validate URL and fetch PDF
    2. Extract text content
    3. Preserve document metadata
    4. Handle errors gracefully
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.document_cache = {}
        logger.debug(f"DocumentProcessor initialized with timeout: {config.PDF_TIMEOUT}s")
    
    def process_document(self, url: str) -> Optional[Dict]:
        """
        Main entry point: Download PDF from URL and extract content
        
        Args:
            url: URL to PDF document
            
        Returns:
            Dictionary with extracted content and metadata
        """
        logger.info(f"Processing document from URL: {url}")
        
        # Validate URL
        if not self.validate_url(url):
            logger.error(f"Invalid URL: {url}")
            return None
        
        # Check cache
        cache_key = self.get_cache_key(url)
        if cache_key in self.document_cache:
            logger.info(f"Document found in cache: {cache_key}")
            return self.document_cache[cache_key]
        
        # Fetch PDF
        pdf_content = self.fetch_pdf(url)
        if not pdf_content:
            return None
        
        # Extract text
        logger.info("Starting text extraction from PDF content")
        extracted = self.extract_text(pdf_content)
        if not extracted:
            logger.error("Failed to extract text from PDF - extraction returned None")
            return None
        
        logger.info(f"Text extraction successful: {extracted['page_count']} pages, {len(extracted['text'])} characters")
        
        # Build result
        result = {
            "url": url,
            "title": self.extract_title(url),
            "content": extracted["text"],
            "pages": extracted["page_count"],
            "metadata": {
                "source_url": url,
                "fetched_at": datetime.now().isoformat(),
                "page_count": extracted["page_count"],
                "content_length": len(extracted["text"]),
                "language": "en"
            },
            "sections": extracted["sections"]  # For semantic chunking
        }
        
        # Cache result
        self.document_cache[cache_key] = result
        
        logger.info(
            f"Document processed successfully: "
            f"{extracted['page_count']} pages, "
            f"{len(extracted['text'])} characters"
        )
        
        return result
    
    def validate_url(self, url: str) -> bool:
        """Validate URL format and accessibility"""
        try:
            parsed = urlparse(url)
            
            # Check URL format
            if not parsed.scheme or not parsed.netloc:
                logger.error(f"Invalid URL format: {url}")
                return False
            
            # Check if PDF
            if not url.lower().endswith('.pdf'):
                logger.warning(f"URL may not be PDF: {url}")
            
            # Try HEAD request to check accessibility
            response = self.session.head(url, allow_redirects=True, timeout=5)
            if response.status_code >= 400:
                logger.error(f"URL not accessible: {url} (Status: {response.status_code})")
                return False
            
            return True
            
        except requests.RequestException as e:
            logger.error(f"URL validation error: {e}")
            return False
    
    def fetch_pdf(self, url: str) -> Optional[bytes]:
        """Download PDF content from URL"""
        try:
            logger.info(f"Fetching PDF from: {url}")
            logger.debug(f"Timeout set to: {config.PDF_TIMEOUT} seconds")
            
            response = self.session.get(
                url,
                timeout=config.PDF_TIMEOUT,
                stream=True,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            response.raise_for_status()
            
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Content-Type: {response.headers.get('content-type', 'Unknown')}")
            
            # Check file size
            content_length = response.headers.get('content-length')
            if content_length:
                content_length_int = int(content_length)
                logger.info(f"Content-Length header: {content_length_int} bytes")
                if content_length_int > config.MAX_PDF_SIZE:
                    logger.error(f"PDF size exceeds limit: {content_length_int} bytes (max: {config.MAX_PDF_SIZE})")
                    return None
            
            pdf_content = response.content
            logger.info(f"Downloaded {len(pdf_content)} bytes")
            
            if len(pdf_content) > config.MAX_PDF_SIZE:
                logger.error(f"Downloaded PDF exceeds size limit: {len(pdf_content)} bytes (max: {config.MAX_PDF_SIZE})")
                return None
            
            if len(pdf_content) < 100:
                logger.warning(f"PDF content seems too small: {len(pdf_content)} bytes")
            
            logger.info(f"PDF fetched successfully: {len(pdf_content)} bytes")
            logger.debug(f"First 100 bytes (hex): {pdf_content[:100].hex()}")
            logger.debug(f"PDF signature check: {pdf_content[:4] == b'%PDF'}")
            
            return pdf_content
            
        except requests.Timeout as e:
            logger.error(f"Timeout while fetching PDF: {e}")
            return None
        except requests.RequestException as e:
            logger.error(f"Error fetching PDF: {e}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching PDF: {e}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return None
    
    def extract_text(self, pdf_content: bytes) -> Optional[Dict]:
        """
        Extract text from PDF content
        Preserves section structure for semantic chunking
        """
        try:
            logger.info(f"Starting PDF text extraction. Content size: {len(pdf_content)} bytes")
            
            if not pdf_content:
                logger.error("PDF content is empty")
                return None
            
            if pdf_content[:4] != b'%PDF':
                logger.error(f"Invalid PDF signature. First 4 bytes: {pdf_content[:4]}")
                return None
            
            logger.debug("Creating BytesIO stream from PDF content")
            pdf_stream = io.BytesIO(pdf_content)
            
            logger.debug("Initializing PdfReader")
            try:
                pdf_reader = pypdf.PdfReader(pdf_stream)
            except Exception as reader_error:
                logger.error(f"Failed to initialize PdfReader: {reader_error}")
                logger.debug(f"PdfReader error details: {traceback.format_exc()}")
                return None
            
            logger.debug("Checking PDF metadata")
            try:
                if pdf_reader.metadata:
                    logger.debug(f"PDF metadata: {pdf_reader.metadata}")
                logger.debug(f"PDF is encrypted: {pdf_reader.is_encrypted}")
                if pdf_reader.is_encrypted:
                    logger.warning("PDF is encrypted, attempting to decrypt")
                    try:
                        pdf_reader.decrypt("")
                        logger.info("PDF decrypted successfully (empty password)")
                    except Exception as decrypt_error:
                        logger.error(f"Failed to decrypt PDF: {decrypt_error}")
                        return None
            except Exception as meta_error:
                logger.warning(f"Could not read PDF metadata: {meta_error}")
            
            page_count = len(pdf_reader.pages)
            logger.info(f"PDF has {page_count} pages")
            
            if page_count == 0:
                logger.error("PDF has no pages")
                return None
            
            # Extract text page by page
            full_text = ""
            sections = []
            pages_with_text = 0
            pages_without_text = 0
            
            logger.debug("Extracting text from pages...")
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    logger.debug(f"Extracting text from page {page_num + 1}/{page_count}")
                    page_text = page.extract_text()
                    
                    if page_text and page_text.strip():
                        pages_with_text += 1
                        text_length = len(page_text.strip())
                        logger.debug(f"Page {page_num + 1}: Extracted {text_length} characters")
                    else:
                        pages_without_text += 1
                        logger.warning(f"Page {page_num + 1}: No text extracted")
                    
                    # Preserve page structure
                    full_text += f"\n--- Page {page_num + 1} ---\n"
                    full_text += page_text if page_text else ""
                    
                    # Try to identify sections (headings, large text)
                    if page_text and page_text.strip():
                        sections.append({
                            "page": page_num + 1,
                            "content": page_text,
                            "heading": self.extract_heading(page_text)
                        })
                except Exception as page_error:
                    logger.error(f"Error extracting text from page {page_num + 1}: {page_error}")
                    logger.debug(f"Page extraction error details: {traceback.format_exc()}")
                    full_text += f"\n--- Page {page_num + 1} ---\n[Error extracting text from this page]"
            
            logger.info(f"Text extraction complete: {pages_with_text} pages with text, {pages_without_text} pages without text")
            logger.info(f"Total extracted text length: {len(full_text)} characters")
            
            if not full_text.strip():
                logger.error("No text extracted from PDF")
                return None
            
            return {
                "text": full_text,
                "page_count": page_count,
                "sections": sections
            }
            
        except pypdf.errors.PdfReadError as e:
            logger.error(f"PDF read error: {e}")
            logger.debug(f"PdfReadError details: {traceback.format_exc()}")
            return None
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return None
    
    def extract_heading(self, page_text: str) -> str:
        """Extract likely heading from page text"""
        lines = page_text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 3 and len(line) < 200:  # Likely a heading
                return line
        return "Unknown Section"
    
    def extract_title(self, url: str) -> str:
        """Extract document title from URL"""
        filename = url.split('/')[-1]
        return filename.replace('.pdf', '').replace('-', ' ').title()
    
    def get_cache_key(self, url: str) -> str:
        """Generate cache key for URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def cleanup_cache(self, keep_recent: int = 10):
        """Keep only recent documents in cache"""
        if len(self.document_cache) > keep_recent:
            keys_to_remove = list(self.document_cache.keys())[:-keep_recent]
            for key in keys_to_remove:
                del self.document_cache[key]
            logger.info(f"Cleaned cache: kept {keep_recent} documents")


# Global document processor instance
document_processor = DocumentProcessor()