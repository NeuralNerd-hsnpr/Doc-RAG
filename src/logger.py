import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any
import json
from pathlib import Path

from config import config


class RequestLogger:
    def __init__(self):
        self.logs_dir = Path(config.LOGS_DIR)
        self.logs_dir.mkdir(exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self):
        log_format = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        
        logging.basicConfig(
            level=getattr(logging, config.LOG_LEVEL),
            format=log_format,
            handlers=[
                logging.FileHandler(
                    self.logs_dir / f"rag_service_{datetime.now().strftime('%Y%m%d')}.log"
                ),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("rag_service")
        self.logger.info("Logging system initialized")
    
    def log_request_start(self, request_id: str, query: str, document_id: Optional[str] = None):
        self.logger.info(
            f"[REQUEST_START] id={request_id} query_length={len(query)} "
            f"document_id={document_id or 'all'}"
        )
    
    def log_router_decision(self, request_id: str, decision: str, confidence: Optional[float] = None):
        self.logger.info(
            f"[ROUTER] id={request_id} decision={decision} "
            f"confidence={confidence or 'N/A'}"
        )
    
    def log_retrieval(
        self, 
        request_id: str, 
        chunks_retrieved: int, 
        similarities: list,
        query_time_ms: float
    ):
        avg_sim = sum(similarities) / len(similarities) if similarities else 0
        self.logger.info(
            f"[RETRIEVAL] id={request_id} chunks={chunks_retrieved} "
            f"avg_similarity={avg_sim:.3f} query_time_ms={query_time_ms:.2f}"
        )
    
    def log_synthesis(
        self,
        request_id: str,
        prompt_length: int,
        response_length: int,
        generation_time_ms: float,
        model: str
    ):
        self.logger.info(
            f"[SYNTHESIS] id={request_id} prompt_len={prompt_length} "
            f"response_len={response_length} time_ms={generation_time_ms:.2f} model={model}"
        )
    
    def log_request_complete(
        self,
        request_id: str,
        total_time_ms: float,
        success: bool,
        error: Optional[str] = None
    ):
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(
            f"[REQUEST_COMPLETE] id={request_id} status={status} "
            f"total_time_ms={total_time_ms:.2f}"
        )
        if error:
            self.logger.error(f"[REQUEST_ERROR] id={request_id} error={error}")
    
    def log_embedding_generation(
        self,
        text_length: int,
        embedding_dim: int,
        generation_time_ms: float
    ):
        self.logger.debug(
            f"[EMBEDDING] text_len={text_length} dim={embedding_dim} "
            f"time_ms={generation_time_ms:.2f}"
        )
    
    def log_chunking(
        self,
        document_id: str,
        pages: int,
        chunks_created: int,
        avg_chunk_size: int,
        chunking_time_ms: float
    ):
        self.logger.info(
            f"[CHUNKING] doc_id={document_id} pages={pages} chunks={chunks_created} "
            f"avg_size={avg_chunk_size} time_ms={chunking_time_ms:.2f}"
        )
    
    def log_vector_storage(
        self,
        document_id: str,
        vectors_stored: int,
        storage_time_ms: float
    ):
        self.logger.info(
            f"[VECTOR_STORAGE] doc_id={document_id} vectors={vectors_stored} "
            f"time_ms={storage_time_ms:.2f}"
        )


request_logger = RequestLogger()

