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
        chunks = []
        content = document["content"]
        
        pages = content.split("--- Page")
        chunk_index = 0
        
        for page_num, page_content in enumerate(pages):
            if page_num > 0:
                page_content = "--- Page" + page_content
            
            page_match = re.search(r'--- Page (\d+)', page_content)
            current_page = int(page_match.group(1)) if page_match else page_num + 1
            
            page_content_clean = re.sub(r'--- Page \d+ ---\s*', '', page_content).strip()
            
            sections = self._split_into_sections(page_content_clean)
            
            for section_text in sections:
                if not section_text.strip():
                    continue
                
                section_token_count = self.count_tokens(section_text)
                
                if section_token_count <= self.chunk_size:
                    if section_token_count >= self.min_chunk_size:
                        chunk = Chunk(
                            content=section_text.strip(),
                            page_number=current_page,
                            section=self.extract_section(section_text),
                            chunk_index=chunk_index,
                            metadata={
                                "strategy": "semantic",
                                "word_count": len(section_text.split()),
                                "token_count": section_token_count,
                                "source": document.get("title", "Unknown"),
                                "chunk_type": "section"
                            }
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                else:
                    sub_chunks = self._split_large_section(section_text, current_page, chunk_index, document)
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
        
        logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    def _split_into_sections(self, text: str) -> List[str]:
        sections = []
        
        heading_patterns = [
            r'^[A-Z][A-Z\s]{3,80}$',
            r'^\d+\.\s+[A-Z][^\n]{3,150}$',
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,10}$',
            r'^[A-Z][A-Z\s]+[?]$',
            r'^[A-Z][a-z]+(?:\s+[a-z]+)*\s+[?]$',
        ]
        
        question_patterns = [
            r'is this',
            r'will.*\?',
            r'what.*\?',
            r'how.*\?',
            r'why.*\?',
        ]
        
        current_section = ""
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                if current_section:
                    current_section += '\n'
                continue
            
            is_heading = False
            
            for pattern in heading_patterns:
                if re.match(pattern, line_stripped):
                    is_heading = True
                    break
            
            if not is_heading:
                for q_pattern in question_patterns:
                    if re.search(q_pattern, line_stripped.lower()):
                        if len(line_stripped) < 100:
                            is_heading = True
                            break
            
            if is_heading and current_section.strip():
                sections.append(current_section.strip())
                current_section = line + '\n'
            else:
                current_section += line + '\n'
        
        if current_section.strip():
            sections.append(current_section.strip())
        
        if not sections:
            sections = [text]
        
        logger.debug(f"[CHUNKING] Split into {len(sections)} sections")
        
        return sections
    
    def _split_large_section(self, text: str, page: int, start_index: int, document: Dict) -> List[Chunk]:
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_idx = start_index
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            test_content = current_chunk + "\n\n" + para if current_chunk else para
            token_count = self.count_tokens(test_content)
            
            if token_count > self.chunk_size and current_chunk:
                if self.count_tokens(current_chunk) >= self.min_chunk_size:
                    chunk_content = current_chunk.strip()
                    
                    if len(chunks) > 0:
                        prev_chunk = chunks[-1]
                        overlap_text = self._get_overlap_text(prev_chunk.content, chunk_content)
                        if overlap_text:
                            chunk_content = overlap_text + "\n\n" + chunk_content
                    
                    chunk = Chunk(
                        content=chunk_content,
                        page_number=page,
                        section=self.extract_section(current_chunk),
                        chunk_index=chunk_idx,
                        metadata={
                            "strategy": "semantic",
                            "word_count": len(chunk_content.split()),
                            "token_count": self.count_tokens(chunk_content),
                            "source": document.get("title", "Unknown"),
                            "chunk_type": "paragraph"
                        }
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
                
                overlap_text = self._get_overlap_text(current_chunk, para)
                current_chunk = overlap_text + "\n\n" + para if overlap_text else para
            else:
                current_chunk = test_content
        
        if current_chunk.strip() and self.count_tokens(current_chunk) >= self.min_chunk_size:
            chunk_content = current_chunk.strip()
            
            if len(chunks) > 0:
                prev_chunk = chunks[-1]
                overlap_text = self._get_overlap_text(prev_chunk.content, chunk_content)
                if overlap_text:
                    chunk_content = overlap_text + "\n\n" + chunk_content
            
            chunk = Chunk(
                content=chunk_content,
                page_number=page,
                section=self.extract_section(current_chunk),
                chunk_index=chunk_idx,
                metadata={
                    "strategy": "semantic",
                    "word_count": len(chunk_content.split()),
                    "token_count": self.count_tokens(chunk_content),
                    "source": document.get("title", "Unknown"),
                    "chunk_type": "paragraph"
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_overlap_text(self, prev_text: str, next_text: str) -> str:
        prev_sentences = prev_text.split('. ')
        if len(prev_sentences) < 2:
            return ""
        
        overlap_sentences = prev_sentences[-2:]
        overlap = '. '.join(overlap_sentences)
        if not overlap.endswith('.'):
            overlap += '.'
        return overlap
    
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
        lines = text.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if not line:
                continue
            
            if len(line) > 3 and len(line) < 150:
                if line.isupper() or (line[0].isupper() and len(line.split()) <= 10):
                    if not line.startswith('---'):
                        return line
        
        if len(text) > 0:
            first_words = ' '.join(text.split()[:5])
            return first_words if len(first_words) < 100 else first_words[:97] + "..."
        
        return "Untitled Section"
    
    def extract_page_number(self, text: str) -> int:
        """Extract page number from text"""
        match = re.search(r'--- Page (\d+)', text)
        if match:
            return int(match.group(1))
        return 1


# Global chunker instance
chunker = DocumentChunker()