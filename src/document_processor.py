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
        self.session.timeout = config.PDF_TIMEOUT
        self.document_cache = {}
    
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
        extracted = self.extract_text(pdf_content)
        if not extracted:
            logger.error("Failed to extract text from PDF")
            return None
        
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
            
            response = self.session.get(
                url,
                timeout=config.PDF_TIMEOUT,
                stream=True
            )
            response.raise_for_status()
            
            # Check file size
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > config.MAX_PDF_SIZE:
                logger.error(f"PDF size exceeds limit: {content_length} bytes")
                return None
            
            pdf_content = response.content
            
            if len(pdf_content) > config.MAX_PDF_SIZE:
                logger.error(f"Downloaded PDF exceeds size limit")
                return None
            
            logger.info(f"PDF fetched successfully: {len(pdf_content)} bytes")
            return pdf_content
            
        except requests.RequestException as e:
            logger.error(f"Error fetching PDF: {e}")
            return None
    
    def extract_text(self, pdf_content: bytes) -> Optional[Dict]:
        """
        Extract text from PDF content
        Preserves section structure for semantic chunking
        """
        try:
            pdf_reader = pypdf.PdfReader(input_pdf_path=pdf_content)
            
            page_count = len(pdf_reader.pages)
            if page_count == 0:
                logger.error("PDF has no pages")
                return None
            
            logger.info(f"PDF has {page_count} pages")
            
            # Extract text page by page
            full_text = ""
            sections = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                
                # Preserve page structure
                full_text += f"\n--- Page {page_num + 1} ---\n"
                full_text += page_text
                
                # Try to identify sections (headings, large text)
                if page_text.strip():
                    sections.append({
                        "page": page_num + 1,
                        "content": page_text,
                        "heading": self.extract_heading(page_text)
                    })
            
            return {
                "text": full_text,
                "page_count": page_count,
                "sections": sections
            }
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
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