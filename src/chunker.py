"""
chunker.py - Document chunking with multiple strategies
Semantic, token-based, and hybrid chunking approaches
"""

import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import tiktoken

from config import config

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a document chunk with metadata"""
    content: str
    page_number: int
    section: str
    chunk_index: int
    metadata: Dict
    
    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "page_number": self.page_number,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata
        }


class DocumentChunker:
    """
    Chunk documents using multiple strategies:
    1. Semantic: Chunk at natural boundaries (sections, paragraphs)
    2. Token-based: Fixed token size with overlap
    3. Hybrid: Combine both approaches
    """
    
    def __init__(self, strategy: str = "semantic"):
        self.strategy = strategy or config.CHUNKING_STRATEGY
        self.chunk_size = config.CHUNK_SIZE
        self.overlap = config.CHUNK_OVERLAP
        self.min_chunk_size = config.MIN_CHUNK_SIZE
        
        # Token counter
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        logger.info(f"Initialized chunker with strategy: {self.strategy}")
    
    def chunk_document(self, document: Dict) -> List[Chunk]:
        """
        Main entry point: Chunk document based on strategy
        
        Args:
            document: Dict with 'content', 'sections', 'title', 'metadata'
            
        Returns:
            List of Chunk objects
        """
        logger.info(f"Chunking document with {self.strategy} strategy")
        
        if self.strategy == "semantic":
            chunks = self.semantic_chunking(document)
        elif self.strategy == "token-based":
            chunks = self.token_based_chunking(document)
        elif self.strategy == "hybrid":
            chunks = self.hybrid_chunking(document)
        else:
            logger.warning(f"Unknown strategy: {self.strategy}, using semantic")
            chunks = self.semantic_chunking(document)
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def semantic_chunking(self, document: Dict) -> List[Chunk]:
        """
        Chunk at semantic boundaries: sections, headings, paragraphs
        Best for structured documents (reports, papers)
        """
        chunks = []
        content = document["content"]
        
        # Split by page breaks first
        pages = content.split("--- Page")
        
        chunk_index = 0
        
        for page_num, page_content in enumerate(pages):
            if page_num > 0:
                # Restore page marker
                page_content = "--- Page" + page_content
            
            # Extract page number
            page_match = re.search(r'--- Page (\d+)', page_content)
            current_page = int(page_match.group(1)) if page_match else page_num + 1
            
            # Split by double newlines (paragraphs)
            paragraphs = page_content.split('\n\n')
            
            current_chunk = ""
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # Check if adding this paragraph exceeds chunk size
                test_content = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
                token_count = self.count_tokens(test_content)
                
                if token_count > self.chunk_size and current_chunk:
                    # Save current chunk
                    if len(current_chunk.strip()) >= self.min_chunk_size:
                        chunk = Chunk(
                            content=current_chunk.strip(),
                            page_number=current_page,
                            section=self.extract_section(current_chunk),
                            chunk_index=chunk_index,
                            metadata={
                                "strategy": "semantic",
                                "word_count": len(current_chunk.split()),
                                "token_count": self.count_tokens(current_chunk),
                                "source": document.get("title", "Unknown")
                            }
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                    
                    # Start new chunk with current paragraph
                    current_chunk = paragraph
                else:
                    # Add to current chunk
                    current_chunk = test_content if current_chunk else paragraph
            
            # Save remaining chunk
            if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
                chunk = Chunk(
                    content=current_chunk.strip(),
                    page_number=current_page,
                    section=self.extract_section(current_chunk),
                    chunk_index=chunk_index,
                    metadata={
                        "strategy": "semantic",
                        "word_count": len(current_chunk.split()),
                        "token_count": self.count_tokens(current_chunk),
                        "source": document.get("title", "Unknown")
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    
    def token_based_chunking(self, document: Dict) -> List[Chunk]:
        """
        Chunk by fixed token size with overlap
        More uniform but may break sentences
        """
        chunks = []
        content = document["content"]
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        chunk_index = 0
        current_chunk = ""
        overlap_buffer = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Add sentence to current chunk
            test_content = current_chunk + " " + sentence if current_chunk else sentence
            token_count = self.count_tokens(test_content)
            
            if token_count > self.chunk_size and current_chunk:
                # Save chunk with overlap
                if len(current_chunk) >= self.min_chunk_size:
                    chunk = Chunk(
                        content=current_chunk.strip(),
                        page_number=self.extract_page_number(current_chunk),
                        section="Content",
                        chunk_index=chunk_index,
                        metadata={
                            "strategy": "token-based",
                            "word_count": len(current_chunk.split()),
                            "token_count": self.count_tokens(current_chunk),
                            "source": document.get("title", "Unknown")
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Create overlap: last sentences of current chunk
                overlap_sentences = overlap_buffer.split()[-20:]  # Last ~20 words
                overlap_buffer = " ".join(overlap_sentences)
                
                # Start new chunk with overlap
                current_chunk = overlap_buffer + " " + sentence
                overlap_buffer = current_chunk
            else:
                current_chunk = test_content
                overlap_buffer = test_content
        
        # Save remaining chunk
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunk = Chunk(
                content=current_chunk.strip(),
                page_number=self.extract_page_number(current_chunk),
                section="Content",
                chunk_index=chunk_index,
                metadata={
                    "strategy": "token-based",
                    "word_count": len(current_chunk.split()),
                    "token_count": self.count_tokens(current_chunk),
                    "source": document.get("title", "Unknown")
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def hybrid_chunking(self, document: Dict) -> List[Chunk]:
        """Combine semantic and token-based chunking"""
        # First do semantic chunking
        semantic_chunks = self.semantic_chunking(document)
        
        # Then split large semantic chunks with token-based approach
        result = []
        for semantic_chunk in semantic_chunks:
            if self.count_tokens(semantic_chunk.content) > self.chunk_size * 1.5:
                # Split this chunk further
                sub_chunks = self.token_based_chunking({
                    "content": semantic_chunk.content,
                    "title": document.get("title", "Unknown")
                })
                result.extend(sub_chunks)
            else:
                result.append(semantic_chunk)
        
        return result
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        try:
            tokens = self.encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}, using word count")
            return len(text.split())
    
    def extract_section(self, text: str) -> str:
        """Extract likely section heading from text"""
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 3 and len(line) < 150 and line.isupper():
                return line
            elif len(line) > 3 and len(line) < 150:
                return line
        return "Untitled Section"
    
    def extract_page_number(self, text: str) -> int:
        """Extract page number from text"""
        match = re.search(r'--- Page (\d+)', text)
        if match:
            return int(match.group(1))
        return 1


# Global chunker instance
chunker = DocumentChunker()