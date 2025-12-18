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
from src.embeddings import embedding_generator

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
            self.dimension = embedding_generator.dimension
            
            logger.info(f"Pinecone client initialized")
            logger.info(f"Expected embedding dimension: {self.dimension}")
            
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
                logger.info(f"Found existing index: {self.index_name}")
                index = self.pc.Index(self.index_name)
                
                try:
                    index_info = self.pc.describe_index(self.index_name)
                    existing_dimension = index_info.dimension
                    
                    if existing_dimension != self.dimension:
                        error_msg = (
                            f"\n{'='*70}\n"
                            f"DIMENSION MISMATCH ERROR\n"
                            f"{'='*70}\n"
                            f"Existing index '{self.index_name}' has dimension: {existing_dimension}\n"
                            f"Current embedding model produces dimension: {self.dimension}\n"
                            f"\nTo fix this, choose one:\n"
                            f"1. Delete the existing index:\n"
                            f"   python scripts/fix_index_dimension.py --delete\n"
                            f"2. Use a different index name (add to .env):\n"
                            f"   PINECONE_INDEX_NAME=document-rag-index-384\n"
                            f"3. Change embedding model to match (1536 dim):\n"
                            f"   EMBEDDING_MODEL=intfloat/e5-large-v2\n"
                            f"{'='*70}\n"
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    
                    logger.info(f"Using existing index: {self.index_name} (dimension: {existing_dimension} ✓)")
                except Exception as e:
                    if "dimension" in str(e).lower() or "mismatch" in str(e).lower():
                        raise
                    logger.warning(f"Could not verify index dimension: {e}")
                
                return index
            
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
        try:
            logger.debug(f"Generating embedding for text (length: {len(text)} chars)")
            embedding = embedding_generator.embed_text(text)
            logger.debug(f"Generated embedding (dimension: {len(embedding)})")
            return embedding
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
                logger.debug(f"Processing chunk {i+1}/{len(chunks)}: page {chunk.page_number}, section {chunk.section}")
                embedding = self.embed_text(chunk.content)
                
                # Create vector ID
                vector_id = f"{document_id}_{chunk.chunk_index}"
                
                # Prepare metadata
                metadata = {
                    "document_id": document_id,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "section": chunk.section,
                    "content": chunk.content,
                    "content_preview": chunk.content[:200],
                    "token_count": chunk.metadata.get("token_count", 0),
                    "word_count": chunk.metadata.get("word_count", 0),
                    "source": chunk.metadata.get("source", "Unknown")
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
        try:
            top_k = top_k or config.RETRIEVAL_TOP_K
            
            query_lower = query.lower()
            is_general = any(word in query_lower for word in [
                "topic", "theme", "subject", "about", "main", "overview", 
                "summary", "what is", "discuss", "cover", "include", "all"
            ])
            
            if is_general:
                expanded_top_k = int(top_k * 4)
                logger.info(f"[RETRIEVAL] General query detected, using expanded_top_k={expanded_top_k}")
            else:
                expanded_top_k = int(top_k * 2.5)
            
            logger.info(f"[RETRIEVAL] Starting retrieval for query: '{query[:100]}...'")
            logger.info(f"[RETRIEVAL] top_k={top_k}, expanded_top_k={expanded_top_k}, threshold={config.SIMILARITY_THRESHOLD}")
            
            from src.query_processor import query_processor
            processed_query = query_processor.preprocess_query(query)
            
            logger.debug(f"[RETRIEVAL] Processed query: '{processed_query}'")
            
            query_embedding = self.embed_text(processed_query)
            logger.debug(f"[RETRIEVAL] Query embedding generated (dim={len(query_embedding)})")
            
            filter_dict = None
            if document_id:
                filter_dict = {"document_id": {"$eq": document_id}}
                logger.debug(f"[RETRIEVAL] Filtering by document_id: {document_id}")
            
            logger.debug(f"[RETRIEVAL] Querying Pinecone with top_k={expanded_top_k}")
            results = self.index.query(
                vector=query_embedding,
                top_k=expanded_top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            logger.info(f"[RETRIEVAL] Pinecone returned {len(results.get('matches', []))} matches")
            
            all_matches = []
            for i, match in enumerate(results.get("matches", [])):
                similarity = match.get("score", 0.0)
                metadata = match.get("metadata", {})
                content = metadata.get("content", metadata.get("content_preview", ""))
                
                logger.debug(
                    f"[RETRIEVAL] Match {i+1}: similarity={similarity:.4f}, "
                    f"page={metadata.get('page_number', 'N/A')}, "
                    f"content_len={len(content)}"
                )
                
                chunk_data = {
                    "id": match["id"],
                    "similarity": similarity,
                    "metadata": metadata,
                    "content": content,
                    "page_number": metadata.get("page_number", 0),
                    "section": metadata.get("section", "Unknown")
                }
                all_matches.append(chunk_data)
            
            logger.info(f"[RETRIEVAL] Total matches before threshold filter: {len(all_matches)}")
            
            if is_general:
                adaptive_threshold = max(config.SIMILARITY_THRESHOLD * 0.7, 0.15)
                logger.info(f"[RETRIEVAL] General query - using adaptive threshold: {adaptive_threshold}")
            else:
                adaptive_threshold = config.SIMILARITY_THRESHOLD
            
            threshold_filtered = [c for c in all_matches if c['similarity'] >= adaptive_threshold]
            logger.info(f"[RETRIEVAL] Matches after threshold ({adaptive_threshold:.3f}): {len(threshold_filtered)}")
            
            if not threshold_filtered and all_matches:
                highest_sim = max(c['similarity'] for c in all_matches)
                logger.warning(
                    f"[RETRIEVAL] All matches below threshold. Highest similarity: {highest_sim:.4f}. "
                    f"Using top {min(top_k * 2, len(all_matches))} matches anyway for better coverage."
                )
                threshold_filtered = sorted(all_matches, key=lambda x: x['similarity'], reverse=True)[:top_k * 2]
            
            retrieved_chunks = self._rerank_chunks(query, threshold_filtered)
            retrieved_chunks = retrieved_chunks[:top_k]
            
            if retrieved_chunks:
                similarities = [c['similarity'] for c in retrieved_chunks]
                min_sim = min(similarities)
                max_sim = max(similarities)
                avg_sim = sum(similarities) / len(similarities)
                
                logger.info(
                    f"[RETRIEVAL] Final result: {len(retrieved_chunks)} chunks "
                    f"(similarity: min={min_sim:.3f}, max={max_sim:.3f}, avg={avg_sim:.3f})"
                )
                
                for i, chunk in enumerate(retrieved_chunks[:3]):
                    logger.debug(
                        f"[RETRIEVAL] Top chunk {i+1}: page={chunk['page_number']}, "
                        f"sim={chunk['similarity']:.3f}, "
                        f"preview={chunk['content'][:100]}..."
                    )
            else:
                logger.warning("[RETRIEVAL] No chunks retrieved after filtering")
                if all_matches:
                    logger.info(f"[RETRIEVAL] Available matches (all below threshold): {len(all_matches)}")
                    logger.info(f"[RETRIEVAL] Highest similarity: {max(c['similarity'] for c in all_matches):.4f}")
            
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"[RETRIEVAL] Error retrieving chunks: {e}", exc_info=True)
            return []
    
    def _rerank_chunks(self, query: str, chunks: List[Dict]) -> List[Dict]:
        if not chunks:
            return chunks
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        query_phrases = []
        words_list = query_lower.split()
        for i in range(len(words_list) - 1):
            query_phrases.append(f"{words_list[i]} {words_list[i+1]}")
        
        logger.debug(f"[RERANK] Reranking {len(chunks)} chunks with query: '{query}'")
        
        for chunk in chunks:
            content = chunk.get("content", "").lower()
            content_words = set(content.split())
            
            word_overlap = len(query_words.intersection(content_words))
            total_query_words = len(query_words)
            
            phrase_matches = sum(1 for phrase in query_phrases if phrase in content)
            
            if total_query_words > 0:
                keyword_score = word_overlap / total_query_words
                phrase_bonus = min(phrase_matches * 0.1, 0.2)
                
                semantic_score = chunk.get("similarity", 0.0)
                
                combined_score = (semantic_score * 0.6) + (keyword_score * 0.3) + phrase_bonus
                chunk["similarity"] = combined_score
                chunk["_original_similarity"] = semantic_score
                chunk["_keyword_score"] = keyword_score
        
        chunks.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        
        logger.debug(
            f"[RERANK] Top 3 after reranking: "
            f"{[(c['similarity'], c.get('_original_similarity', 0)) for c in chunks[:3]]}"
        )
        
        return chunks
    
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