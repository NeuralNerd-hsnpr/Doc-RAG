"""
config.py - Configuration Management for Document RAG System
Handles environment variables, API keys, and system settings
"""

import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


class Config:
    """Base configuration class"""
    
    # Anthropic API
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = "claude-opus-4-1"
    
    # Pinecone Vector DB
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "document-rag-index")
    PINECONE_DIMENSION: int = 1536  # OpenAI/Anthropic embedding dimension
    
    # Document Processing
    CHUNK_SIZE: int = 800  # tokens per chunk
    CHUNK_OVERLAP: int = 100  # overlap for context preservation
    MIN_CHUNK_SIZE: int = 100  # minimum tokens in chunk
    
    # Chunking strategy
    CHUNKING_STRATEGY: str = "semantic"  # "semantic", "token-based", or "hybrid"
    
    # Embedding
    EMBEDDING_MODEL: str = "text-embedding-3-large"  # For API-based embeddings
    EMBEDDING_DIMENSION: int = 1536
    
    # Retrieval
    RETRIEVAL_TOP_K: int = 5  # number of chunks to retrieve
    SIMILARITY_THRESHOLD: float = 0.75  # minimum cosine similarity
    
    # LLM Synthesis
    MAX_TOKENS_RESPONSE: int = 2000
    TEMPERATURE: float = 0.2  # Lower for factual answers
    
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
        required = ["ANTHROPIC_API_KEY", "PINECONE_API_KEY"]
        missing = [key for key in required if not getattr(self, key)]
        
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                f"Please set them in .env file"
            )
    
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
