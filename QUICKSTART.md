# Quick Start Guide

## Prerequisites

1. Python 3.10+
2. Hugging Face API token ([Get one here](https://huggingface.co/settings/tokens))
3. Pinecone API key ([Sign up here](https://app.pinecone.io/))

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys:
# HF_API_TOKEN=hf_...
# PINECONE_API_KEY=...
# PINECONE_ENVIRONMENT=us-east-1
```

## Running the Service

### Option 1: CLI Interface (Original)

```bash
python main.py
```

Then follow the interactive menu to:
1. Ingest a document from URL
2. Ask questions about the document

### Option 2: API Server (New)

```bash
# Start the API server
python api.py

# Or using uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Usage Examples

### 1. Health Check

```bash
curl http://localhost:8000/health
```

### 2. Ingest Document

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/document.pdf"}'
```

### 3. Query Document

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main themes?",
    "document_id": "doc_20240115_103000"
  }'
```

### 4. Get Statistics

```bash
curl http://localhost:8000/api/v1/stats
```

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Ingest document
response = requests.post(
    f"{BASE_URL}/api/v1/ingest",
    json={"url": "https://example.com/document.pdf"}
)
result = response.json()
document_id = result["document_id"]
print(f"Document ingested: {document_id}")

# Query document
response = requests.post(
    f"{BASE_URL}/api/v1/query",
    json={
        "query": "What is the main topic?",
        "document_id": document_id
    }
)
result = response.json()
print(f"Answer: {result['answer']}")
print(f"Citations: {result['citations']}")
```

## Configuration

Edit `.env` file to customize:

```bash
# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# LLM Model
HF_MODEL=HuggingFaceH4/zephyr-7b-beta

# Chunking
CHUNK_SIZE=800
CHUNK_OVERLAP=100

# Retrieval
RETRIEVAL_TOP_K=5
SIMILARITY_THRESHOLD=0.75

# Logging
LOG_LEVEL=INFO
```

## Logging

Logs are written to:
- Console (stdout)
- File: `logs/rag_service_YYYYMMDD.log`

Each request is traced with:
- Request ID
- Timing metrics
- Performance data
- Error details

## Troubleshooting

### "HF_API_TOKEN not found"
- Check your `.env` file
- Verify token starts with `hf_`
- Get token from: https://huggingface.co/settings/tokens

### "Pinecone connection failed"
- Verify `PINECONE_API_KEY` in `.env`
- Check Pinecone dashboard: https://app.pinecone.io/

### "Embedding generation failed"
- Check Hugging Face API status
- Verify model name is correct
- Check API token permissions

### "No chunks retrieved"
- Verify document was ingested successfully
- Check similarity threshold (try lowering it)
- Ensure document_id matches

## Performance Tips

1. **Use appropriate chunk size:** Smaller chunks (500-800 tokens) work better for most documents
2. **Adjust similarity threshold:** Lower threshold (0.6-0.7) for broader results
3. **Increase top_k:** For complex queries, increase `RETRIEVAL_TOP_K` to 7-10
4. **Monitor logs:** Check `logs/` directory for performance metrics

## Next Steps

- Read `README_API.md` for detailed API documentation
- Read `IMPROVEMENTS.md` for technical details
- Check logs for debugging information

