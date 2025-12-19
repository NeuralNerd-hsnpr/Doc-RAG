from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import logging
import uuid
from datetime import datetime

from config import config
from src.document_processor import document_processor
from src.chunker import chunker
from src.vector_store import vector_store
from src.langgraph_workflow import rag_workflow
from src.logger import request_logger

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document RAG Service",
    description="Production-ready Document RAG agent service",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DocumentIngestRequest(BaseModel):
    url: str = Field(..., description="URL of the PDF document to ingest")
    document_id: Optional[str] = Field(None, description="Optional custom document ID")


class DocumentIngestResponse(BaseModel):
    success: bool
    document_id: str
    message: str
    metadata: Dict


class QueryRequest(BaseModel):
    query: str = Field(..., description="Question to ask about the document")
    document_id: Optional[str] = Field(None, description="Optional document ID to search within")


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[Dict]
    router_decision: str
    chunks_retrieved: int
    execution_time_seconds: float
    request_id: str
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


@app.get("/", response_model=Dict)
async def root():
    return {
        "service": "Document RAG Service",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "ingest": "/api/v1/ingest",
            "query": "/api/v1/query",
            "stats": "/api/v1/stats"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0"
    )


@app.post("/api/v1/ingest", response_model=DocumentIngestResponse)
async def ingest_document(request: DocumentIngestRequest):
    try:
        logger.info(f"Ingesting document from URL: {request.url}")
        
        document = document_processor.process_document(request.url)
        if not document:
            raise HTTPException(status_code=400, detail="Failed to process document")
        
        chunks = chunker.chunk_document(document)
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to chunk document")
        
        document_id = request.document_id or f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        success = vector_store.store_chunks(chunks, document_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store vectors")
        
        metadata = {
            "url": request.url,
            "title": document.get("title", "Unknown"),
            "pages": document.get("pages", 0),
            "chunks": len(chunks),
            "ingested_at": datetime.now().isoformat()
        }
        
        logger.info(f"Document ingested successfully: {document_id}")
        
        return DocumentIngestResponse(
            success=True,
            document_id=document_id,
            message=f"Document ingested successfully: {len(chunks)} chunks stored",
            metadata=metadata
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    try:
        request_id = str(uuid.uuid4())
        logger.info(f"Processing query: {request.query[:100]}...")
        
        result = rag_workflow.process_query(
            query=request.query,
            document_id=request.document_id
        )
        
        result["request_id"] = request_id
        
        return QueryResponse(**result)
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/api/v1/stats")
async def get_stats():
    try:
        stats = vector_store.get_index_stats()
        return {
            "index_name": vector_store.index_name,
            "dimension": vector_store.dimension,
            "total_vectors": stats.get("total_vector_count", 0),
            "namespaces": stats.get("namespaces", {})
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.delete("/api/v1/documents/{document_id}")
async def delete_document(document_id: str):
    try:
        success = vector_store.delete_document_vectors(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found or deletion failed")
        
        return {"success": True, "message": f"Document {document_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

