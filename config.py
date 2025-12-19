"""
config.py - Configuration Management for Document RAG System
Handles environment variables, API keys, and system settings
"""

import os
import logging
from dotenv import load_dotenv
from typing import Optional

load_dotenv()
logger = logging.getLogger(__name__)


class Config:
    """Base configuration class"""
    
    # Hugging Face API
    HF_API_TOKEN: str = (os.getenv("HF_API_TOKEN", "") or os.getenv("HF_API_KEY", "")).strip()
    HF_MODEL: str = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta").strip()
    
    # Pinecone Vector DB
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "document-rag-index")
    PINECONE_DIMENSION: int = 384
    
    # Document Processing
    CHUNK_SIZE: int = 800  # tokens per chunk
    CHUNK_OVERLAP: int = 100  # overlap for context preservation
    MIN_CHUNK_SIZE: int = 100  # minimum tokens in chunk
    
    # Chunking strategy
    CHUNKING_STRATEGY: str = "semantic"  # "semantic", "token-based", or "hybrid"
    
    # Embedding
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = 384
    
    # Retrieval
    RETRIEVAL_TOP_K: int = 10  # number of chunks to retrieve
    SIMILARITY_THRESHOLD: float = 0.3  # minimum cosine similarity (lowered for better recall)
    USE_QUERY_EXPANSION: bool = True  # expand queries for better matching
    
    # LLM Synthesis
    MAX_TOKENS_RESPONSE: int = 600  # Reduced to prevent repetition loops
    TEMPERATURE: float = 0.4  # Higher to break deterministic loops
    REPETITION_PENALTY: float = 1.15  # Penalty to prevent repetition loops
    
    # File Management
    DATA_DIR: str = "./data"
    LOGS_DIR: str = "./logs"
    CACHE_DIR: str = "./cache"
    
    # PDF Processing
    PDF_TIMEOUT: int = 30  # seconds to fetch PDF
    MAX_PDF_SIZE: int = 50 * 1024 * 1024  # 50MB max
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __init__(self):
        """Validate configuration on initialization"""
        self.validate()
    
    def validate(self):
        """Validate that all required configs are set"""
        if not self.HF_API_TOKEN:
            raise ValueError(
                "HF_API_TOKEN or HF_API_KEY is required. Please set one in your .env file.\n"
                "Get your token from: https://huggingface.co/settings/tokens\n"
                "Use either: HF_API_TOKEN=hf_... or HF_API_KEY=hf_..."
            )
        
        required = ["PINECONE_API_KEY"]
        missing = [key for key in required if not getattr(self, key)]
        
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                f"Please set them in .env file"
            )
        
        if not self.HF_API_TOKEN.startswith("hf_"):
            logger.warning(
                f"HF_API_TOKEN format may be incorrect. "
                f"Expected to start with 'hf_', got: {self.HF_API_TOKEN[:10]}..."
            )
        else:
            logger.info(f"HF_API_TOKEN loaded (length: {len(self.HF_API_TOKEN)})")
        
        if self.PINECONE_API_KEY:
            logger.info(f"PINECONE_API_KEY loaded (length: {len(self.PINECONE_API_KEY)})")
        
        logger.info(f"Using Hugging Face model: {self.HF_MODEL}")
        logger.info("Using HF Inference API only (no local models)")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return os.getenv("ENVIRONMENT") == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return os.getenv("ENVIRONMENT") in [None, "development"]


class DevelopmentConfig(Config):
    """Development-specific configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    RETRIEVAL_TOP_K = 3
    CHUNK_SIZE = 500


class ProductionConfig(Config):
    """Production-specific configuration"""
    DEBUG = False
    LOG_LEVEL = "WARNING"
    RETRIEVAL_TOP_K = 7
    CHUNK_SIZE = 1000


def get_config() -> Config:
    """Factory function to get appropriate config"""
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        return ProductionConfig()
    return DevelopmentConfig()


# Global config instance
config = get_config()
