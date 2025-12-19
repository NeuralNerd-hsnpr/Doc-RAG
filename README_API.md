# Document RAG Service API Documentation

## Overview

The Document RAG Service provides a RESTful API for document ingestion and question-answering using Retrieval-Augmented Generation (RAG).

## Base URL

```
http://localhost:8000
```

## Endpoints

### Health Check

**GET** `/health`

Check service health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "version": "2.0.0"
}
```

### Ingest Document

**POST** `/api/v1/ingest`

Ingest a PDF document from a URL.

**Request Body:**
```json
{
  "url": "https://example.com/document.pdf",
  "document_id": "optional_custom_id"
}
```

**Response:**
```json
{
  "success": true,
  "document_id": "doc_20240115_103000",
  "message": "Document ingested successfully: 156 chunks stored",
  "metadata": {
    "url": "https://example.com/document.pdf",
    "title": "Document Title",
    "pages": 50,
    "chunks": 156,
    "ingested_at": "2024-01-15T10:30:00"
  }
}
```

### Query Document

**POST** `/api/v1/query`

Ask a question about ingested documents.

**Request Body:**
```json
{
  "query": "What are the main themes discussed?",
  "document_id": "doc_20240115_103000"
}
```

**Response:**
```json
{
  "question": "What are the main themes discussed?",
  "answer": "The document discusses three main themes...",
  "citations": [
    {
      "section": "1",
      "page": 5,
      "topic": "Introduction",
      "similarity_score": 0.891,
      "source_document": "Document Title"
    }
  ],
  "router_decision": "summary",
  "chunks_retrieved": 6,
  "execution_time_seconds": 3.45,
  "request_id": "uuid-here",
  "error": null
}
```

### Get Statistics

**GET** `/api/v1/stats`

Get vector store statistics.

**Response:**
```json
{
  "index_name": "document-rag-index",
  "dimension": 384,
  "total_vectors": 156,
  "namespaces": {}
}
```

### Delete Document

**DELETE** `/api/v1/documents/{document_id}`

Delete a document and all its vectors.

**Response:**
```json
{
  "success": true,
  "message": "Document doc_20240115_103000 deleted successfully"
}
```

## Usage Examples

### Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Ingest document
response = requests.post(
    f"{BASE_URL}/api/v1/ingest",
    json={"url": "https://example.com/document.pdf"}
)
doc_id = response.json()["document_id"]

# Query document
response = requests.post(
    f"{BASE_URL}/api/v1/query",
    json={
        "query": "What is the main topic?",
        "document_id": doc_id
    }
)
answer = response.json()["answer"]
print(answer)
```

### cURL

```bash
# Ingest document
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/document.pdf"}'

# Query document
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "document_id": "doc_20240115_103000"}'
```

## Running the API

```bash
# Using uvicorn directly
uvicorn api:app --host 0.0.0.0 --port 8000

# Or using Python
python api.py
```

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found (document not found)
- `500`: Internal Server Error

Error responses include a `detail` field with error information:

```json
{
  "detail": "Failed to process document"
}
```

