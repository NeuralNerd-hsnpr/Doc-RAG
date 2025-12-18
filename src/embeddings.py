import logging
import os
from typing import List, Optional
from huggingface_hub import InferenceClient
from config import config

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    def __init__(self):
        self.model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.dimension = self._get_model_dimension()
        self.client = None
        self._initialize_client()
    
    def _get_model_dimension(self) -> int:
        model_dimensions = {
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "intfloat/e5-large-v2": 1024,
            "BAAI/bge-large-en-v1.5": 1024,
            "BAAI/bge-base-en-v1.5": 768,
        }
        dimension = model_dimensions.get(self.model_name, config.PINECONE_DIMENSION)
        config.PINECONE_DIMENSION = dimension
        return dimension
    
    def _initialize_client(self):
        try:
            hf_token = config.HF_API_TOKEN
            if not hf_token:
                raise ValueError("HF_API_TOKEN is required for embeddings")
            
            self.client = InferenceClient(
                model=self.model_name,
                token=hf_token
            )
            logger.info(f"Embedding client initialized with model: {self.model_name}")
            logger.info(f"Embedding dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding client: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * self.dimension
        
        try:
            import time
            start_time = time.time()
            
            response = self.client.feature_extraction(text)
            
            generation_time = (time.time() - start_time) * 1000
            try:
                from src.logger import request_logger
                request_logger.log_embedding_generation(len(text), self.dimension, generation_time)
            except (ImportError, AttributeError):
                pass
            
            if isinstance(response, list):
                if len(response) > 0 and isinstance(response[0], list):
                    embedding = response[0]
                else:
                    embedding = response
            elif hasattr(response, 'tolist'):
                embedding = response.tolist()
            elif hasattr(response, '__iter__'):
                embedding = list(response)
            else:
                logger.error(f"Unexpected response type: {type(response)}")
                return [0.0] * self.dimension
            
            if not isinstance(embedding, list):
                logger.error(f"Embedding is not a list: {type(embedding)}")
                return [0.0] * self.dimension
            
            if len(embedding) != self.dimension:
                logger.warning(
                    f"Embedding dimension mismatch: expected {self.dimension}, "
                    f"got {len(embedding)}. Padding or truncating."
                )
                if len(embedding) < self.dimension:
                    embedding.extend([0.0] * (self.dimension - len(embedding)))
                else:
                    embedding = embedding[:self.dimension]
            
            return [float(x) for x in embedding]
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = [self.embed_text(text) for text in batch]
                embeddings.extend(batch_embeddings)
                logger.debug(f"Embedded batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error embedding batch {i//batch_size + 1}: {e}")
                for text in batch:
                    embeddings.append([0.0] * self.dimension)
        return embeddings


embedding_generator = EmbeddingGenerator()

