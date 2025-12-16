"""
vector_store.py - Pinecone vector database integration
Handles vector storage, retrieval, and index management
"""

import logging
from typing import List, Dict, Tuple, Optional
import json
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeApiException
from config import config
from src.chunker import Chunk

logger = logging.getLogger(__name__)


class PineconeVectorStore:
    """
    Manage vectors in Pinecone:
    1. Create/initialize index
    2. Embed chunks and store vectors
    3. Retrieve similar chunks
    4. Manage metadata
    """
    
    def __init__(self):
        """Initialize Pinecone client"""
        try:
            self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
            self.index_name = config.PINECONE_INDEX_NAME
            self.dimension = config.PINECONE_DIMENSION
            
            logger.info(f"Pinecone client initialized")
            
            # Get or create index
            self.index = self.get_or_create_index()
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise
    
    def get_or_create_index(self):
        """Get existing index or create new one"""
        try:
            existing_indexes = self.pc.list_indexes()
            index_names = []
            
            if existing_indexes:
                if isinstance(existing_indexes, list):
                    if existing_indexes and hasattr(existing_indexes[0], 'name'):
                        index_names = [idx.name for idx in existing_indexes]
                    else:
                        index_names = [str(idx) for idx in existing_indexes]
                elif hasattr(existing_indexes, '__iter__'):
                    try:
                        index_names = [idx.name if hasattr(idx, 'name') else str(idx) for idx in existing_indexes]
                    except:
                        index_names = list(existing_indexes) if isinstance(existing_indexes, (list, tuple)) else []
            
            if self.index_name in index_names:
                logger.info(f"Using existing index: {self.index_name}")
                return self.pc.Index(self.index_name)
            
            logger.info(f"Creating new index: {self.index_name}")
            try:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=config.PINECONE_ENVIRONMENT
                    )
                )
                logger.info(f"Index {self.index_name} created successfully")
            except PineconeApiException as create_error:
                error_str = str(create_error)
                if "ALREADY_EXISTS" in error_str or "409" in error_str or "already exists" in error_str.lower():
                    logger.info(f"Index {self.index_name} already exists (detected during creation), using it")
                else:
                    raise
            
            return self.pc.Index(self.index_name)
                
        except Exception as e:
            error_str = str(e)
            if "ALREADY_EXISTS" in error_str or "409" in error_str or "already exists" in error_str.lower():
                logger.info(f"Index {self.index_name} already exists, using it")
                return self.pc.Index(self.index_name)
            logger.error(f"Error getting/creating index: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for text
        Using hash-based embeddings (placeholder - can be replaced with HF embeddings)
        """
        try:
            # Placeholder: hash-based embeddings
            # In production, you could use:
            # from sentence_transformers import SentenceTransformer
            # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            # return model.encode(text).tolist()
            
            # Placeholder: create deterministic embedding from text
            import hashlib
            import numpy as np
            
            # Create seed from text
            hash_obj = hashlib.sha256(text.encode())
            seed = int(hash_obj.hexdigest(), 16) % (2**32)
            
            # Generate deterministic random embedding
            np.random.seed(seed)
            embedding = np.random.randn(self.dimension).astype('float32')
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def store_chunks(self, chunks: List[Chunk], document_id: str) -> bool:
        """
        Store document chunks in Pinecone
        
        Args:
            chunks: List of Chunk objects to store
            document_id: Unique identifier for source document
            
        Returns:
            bool indicating success
        """
        try:
            logger.info(f"Storing {len(chunks)} chunks for document {document_id}")
            
            vectors_to_upsert = []
            
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.embed_text(chunk.content)
                
                # Create vector ID
                vector_id = f"{document_id}_{chunk.chunk_index}"
                
                # Prepare metadata
                metadata = {
                    "document_id": document_id,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "section": chunk.section,
                    "content_preview": chunk.content[:200],  # Preview for debugging
                    **chunk.metadata
                }
                
                # Add to batch
                vectors_to_upsert.append((
                    vector_id,
                    embedding,
                    metadata
                ))
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.info(f"Upserted batch {i//batch_size + 1}")
            
            logger.info(f"Successfully stored {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error storing chunks: {e}")
            return False
    
    def retrieve_relevant_chunks(
        self, 
        query: str, 
        document_id: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: Search query
            document_id: Optional filter by document
            top_k: Number of results (default from config)
            
        Returns:
            List of relevant chunks with scores
        """
        try:
            top_k = top_k or config.RETRIEVAL_TOP_K
            
            # Generate query embedding
            query_embedding = self.embed_text(query)
            
            # Build filter if document_id specified
            filter_dict = None
            if document_id:
                filter_dict = {"document_id": {"$eq": document_id}}
            
            # Query the index
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Format results
            retrieved_chunks = []
            for match in results["matches"]:
                retrieved_chunks.append({
                    "id": match["id"],
                    "similarity": match["score"],
                    "metadata": match["metadata"],
                    # Note: we'd need to fetch content separately
                    # Pinecone doesn't store full content in default setup
                })
            
            if retrieved_chunks:
                min_similarity = min([c['similarity'] for c in retrieved_chunks])
                logger.info(
                    f"Retrieved {len(retrieved_chunks)} chunks "
                    f"(min similarity: {min_similarity:.4f})"
                )
            else:
                logger.warning("No chunks retrieved from vector store")
            
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def get_chunk_content(self, chunk_id: str) -> Optional[str]:
        """
        Get full content of a chunk
        In production, store full content in separate DB or retrieve from cache
        """
        try:
            # This is a simplified version
            # In production, you'd:
            # 1. Store full content in separate database (PostgreSQL, MongoDB)
            # 2. Or cache it with the metadata
            # 3. Or retrieve from original document
            
            # For now, extract from metadata preview and reconstruct
            # (This is limited and should be improved)
            
            logger.warning(f"get_chunk_content not fully implemented for {chunk_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting chunk content: {e}")
            return None
    
    def delete_document_vectors(self, document_id: str) -> bool:
        """Delete all vectors for a document"""
        try:
            logger.info(f"Deleting vectors for document {document_id}")
            
            # Query to find all vectors for this document
            filter_dict = {"document_id": {"$eq": document_id}}
            
            # Get vector IDs to delete
            results = self.index.query(
                vector=[0.0] * self.dimension,  # Dummy query
                filter=filter_dict,
                top_k=10000  # Get all vectors for this document
            )
            
            # Delete them
            ids_to_delete = [match["id"] for match in results["matches"]]
            if ids_to_delete:
                self.index.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} vectors")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document vectors: {e}")
            return False
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the index"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "namespaces": stats.namespaces,
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}


# Global vector store instance
vector_store = PineconeVectorStore()