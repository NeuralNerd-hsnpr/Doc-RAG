# Document RAG System - Production-Ready Implementation

A complete, production-grade Retrieval-Augmented Generation (RAG) system for intelligent document Q&A with PDF ingestion, vector embeddings, and precise answer synthesis.

**Key Features:**
-  **Dynamic Document Ingestion**: Load PDFs from URLs automatically
-  **Intelligent Chunking**: Semantic, token-based, and hybrid strategies
-  **Vector Storage**: Pinecone integration for scalable similarity search
-  **LLM-Powered Answers**: Claude API for precise, cited responses
-  **LangGraph Workflow**: Structured RAG pipeline with routing, retrieval, synthesis
-  **Interactive CLI**: User-friendly question answering interface
-  **Citation Tracking**: All answers grounded in source documents
-  **Production-Ready**: Error handling, logging, configuration management

---

##  Quick Start 

### Prerequisites
- Python 3.10+
- Anthropic API key ([Get one here](https://console.anthropic.com/))
- Pinecone account ([Sign up here](https://app.pinecone.io/))

### Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/document-rag-system.git
cd document-rag-system

# 2. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure credentials
cp .env.example .env
# Edit .env with your API keys

# 5. Run the system
python main.py
```

### First Use
1. When prompted, paste a document URL (any PDF)
2. System automatically downloads, chunks, and vectorizes it
3. Ask questions and get answers with citations!

Example:
```
Enter document URL: https://example.com/document.pdf
→ Document processed (50 pages, 5000 chunks)
→ Vectors stored in Pinecone

Ask a question: What are the main topics covered?
Answer: [Generated from document content]
Citations: [Section 1, Page 5], [Section 3, Page 12]
```

---

##  Installation Guide (Step by Step)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/document-rag-system.git
cd document-rag-system
```

### Step 2: Set Up Python Environment
```bash
# Create virtual environment
python3.10 -m venv venv

# Activate it
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# Verify activation (should show "venv" prefix)
which python
```

### Step 3: Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# Verify installation
python -c "import anthropic, pinecone; print('✓ Ready')"
```

### Step 4: Configure Credentials

**Get Anthropic API Key:**
1. Visit https://console.anthropic.com/
2. Create/log in to account
3. Go to API Keys section
4. Create new API key
5. Copy the key (starts with `sk-ant-`)

**Get Pinecone Credentials:**
1. Visit https://app.pinecone.io/
2. Create/log in to account
3. Create new project
4. Go to API Keys section
5. Copy API Key and Environment name

**Set Environment Variables:**
```bash
# Copy example file
cp .env.example .env

# Edit with your credentials (use your favorite editor)
nano .env

# Add these three lines:
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
PINECONE_API_KEY=xxxxxxxxxxxxx
PINECONE_ENVIRONMENT=us-east-1
```

### Step 5: Verify Setup
```bash
# Test configuration
python -c "from config import config; print('✓ Config OK')"

# Test Pinecone connection
python -c "from src.vector_store import vector_store; print('✓ Pinecone OK')"
```

### Step 6: Run the System
```bash
python main.py
```

You should see the interactive menu!

---

## 🎯 How to Use

### Main Menu Options

```
MAIN MENU
1. Ingest Document (from URL)
2. Ask Question (about current document)
3. Show Document Info
4. Show Vector Store Stats
5. Delete Document from Index
6. Exit
```

### Workflow Example

```
$ python main.py

[MAIN MENU]
Select option (1-6): 1

[STEP 1: DOCUMENT INGESTION]
Enter document URL: https://www.jpmorgan.com/content/dam/jpmorgan/documents/outlook.pdf

→ Downloading and extracting PDF...
✓ Document processed:
  Title: Outlook 2025 Building on Strength
  Pages: 80
  Content length: 450000 characters

→ Chunking document...
✓ Created 156 chunks

→ Generating embeddings and storing in Pinecone...
✓ Vectors stored successfully
  Document ID: doc_20240115_143022

[MAIN MENU]
Select option (1-6): 2

[STEP 2: Q&A]
Document: Outlook 2025 Building on Strength
Document ID: doc_20240115_143022

Enter your question: What are the main investment themes for 2025?

Processing: What are the main investment themes for 2025?
Generating answer...

ANSWER
================================================================================

According to the document, the main investment themes for 2025 are:

1. Easing Global Policy - Policy rates normalizing and economies expanding
2. Accelerating Capital Investment - Spending on AI, power, infrastructure, security
3. Understanding Election Impacts - Less regulation, wider deficits, more tariffs
4. Renewing Portfolio Resilience - Focus on income and real assets
5. Evolving Investment Landscapes - Opportunities in alternatives, sports, cities

[CITATIONS]
[SECTION 1]
  Page: 3
  Topic: Key Takeaways
  Relevance Score: 0.891

Processing Time: 2.34s
```

---

## 📂 Project Structure

```
document-rag-system/
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup configuration
│
├── config.py                    # Configuration management
├── main.py                      # Interactive CLI entry point
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── document_processor.py    # PDF download & extraction
│   ├── chunker.py              # Document chunking strategies
│   ├── vector_store.py         # Pinecone integration
│   ├── langgraph_workflow.py   # LangGraph RAG pipeline
│   ├── embeddings.py           # Embedding generation
│   ├── retriever.py            # Vector similarity search
│   └── utils.py                # Utility functions
│
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── test_document_processor.py
│   ├── test_chunker.py
│   └── test_retriever.py
│
├── data/                        # Downloaded PDFs storage
│   └── .gitkeep
│
└── logs/                        # Application logs
    └── .gitkeep
```

---

## 🔧 Configuration Options

Edit `.env` to customize behavior:

### Document Processing
```bash
CHUNK_SIZE=800              # Tokens per chunk
CHUNK_OVERLAP=100           # Overlap between chunks
MIN_CHUNK_SIZE=100          # Minimum chunk size
CHUNKING_STRATEGY=semantic  # semantic|token-based|hybrid
```

### Vector Database
```bash
PINECONE_INDEX_NAME=document-rag-index
PINECONE_DIMENSION=1536  # Embedding dimension
RETRIEVAL_TOP_K=5        # Documents to retrieve
SIMILARITY_THRESHOLD=0.75 # Minimum relevance score
```

### LLM Configuration
```bash
ANTHROPIC_MODEL=claude-opus-4-1
TEMPERATURE=0.2              # Lower = more factual
MAX_TOKENS_RESPONSE=2000     # Max answer length
```

### System
```bash
ENVIRONMENT=development     # development|production
LOG_LEVEL=INFO             # DEBUG|INFO|WARNING|ERROR
```

---

## 🔄 System Architecture

### 1. **Document Ingestion Pipeline**
```
URL → Download PDF → Extract Text → Preserve Metadata
```

### 2. **Chunking Strategies**
- **Semantic**: Chunk at section/paragraph boundaries (best for structured docs)
- **Token-based**: Fixed token size with overlap (uniform chunks)
- **Hybrid**: Semantic + token-based (best of both)

### 3. **LangGraph Workflow**
```
Query → Router → Retrieval → Synthesis → Formatter → Answer
  ↓         ↓        ↓          ↓         ↓
Classify  Query   Pinecone  Claude   Extract
Intent    Type    Vectors   Answer   Citations
```

### 4. **Vector Storage & Retrieval**
```
Chunks → Embed → Pinecone → Semantic Search → Top-K Results
```

### 5. **Answer Generation**
```
Retrieved Chunks → Claude LLM → Grounded Answer with Citations
```

---

##  Testing

Run unit tests:
```bash
pytest tests/ -v

# Run specific test
pytest tests/test_chunker.py -v

# Run with coverage
pytest tests/ --cov=src
```

Test individual modules:
```bash
# Test document processor
python -c "from src.document_processor import DocumentProcessor; dp = DocumentProcessor(); print('✓ OK')"

# Test chunker
python -c "from src.chunker import DocumentChunker; dc = DocumentChunker(); print('✓ OK')"

# Test vector store
python -c "from src.vector_store import PineconeVectorStore; ps = PineconeVectorStore(); print('✓ OK')"
```

---

##  Troubleshooting

### "ModuleNotFoundError: No module named 'X'"
```bash
# Solution: Install missing package
pip install -r requirements.txt

# Or install specific package
pip install anthropic pinecone-client langgraph
```

### "ANTHROPIC_API_KEY not configured"
```bash
# Check .env file exists
ls -la .env

# Verify API key is set
grep ANTHROPIC_API_KEY .env

# If missing, add it:
echo "ANTHROPIC_API_KEY=sk-ant-xxxxx" >> .env
```

### "Failed to connect to Pinecone"
```bash
# Verify credentials
grep PINECONE .env

# Test connection
python -c "from pinecone import Pinecone; print(Pinecone(api_key='YOUR_KEY'))"

# Check Pinecone dashboard for status
# Visit: https://app.pinecone.io/
```

### "PDF extraction failed"
```bash
# Ensure URL is valid and accessible
curl -I https://your-pdf-url.pdf

# Check file size
# Max size: 50MB (adjust in config.py if needed)

# Verify PDF is not encrypted
# Try opening in browser
```

### "Out of memory during chunking"
```bash
# Reduce CHUNK_SIZE in .env
CHUNK_SIZE=500  # Default is 800

# Or process in smaller batches
# Modify document_processor.py to handle streaming
```

---

##  Performance Metrics

### Ingestion Performance
- PDF download: 1-5 seconds (depends on file size)
- Text extraction: 2-10 seconds (depends on pages)
- Chunking: 1-3 seconds (156 chunks)
- Embedding: 5-15 seconds (vector generation)
- **Total ingestion: ~15-30 seconds**

### Q&A Performance
- Query routing: <100ms
- Vector retrieval: 200-500ms (Pinecone latency)
- Answer synthesis: 2-5 seconds (Claude API)
- **Total Q&A: ~3-6 seconds**

### Scalability
- Max document size: 50 MB
- Max chunks per document: 10,000+
- Pinecone supports millions of vectors
- Linear scaling with document count

---

##  Deployment Options

### Local Development
```bash
python main.py
# Simple development setup
```

### Docker Deployment
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

### Cloud Deployment (AWS Lambda)
```python
# Handler for AWS Lambda
def lambda_handler(event, context):
    question = event['query']
    document_id = event['document_id']
    result = rag_workflow.process_query(question, document_id)
    return result
```

### Serverless (Google Cloud Functions)
Similar to Lambda - wrap main.process_query() as HTTP endpoint

---

## 📈 Production Checklist

- [ ] Set `ENVIRONMENT=production` in .env
- [ ] Enable logging to CloudWatch/ELK
- [ ] Set up error tracking (Sentry)
- [ ] Configure Pinecone backup/replication
- [ ] Set up monitoring and alerts
- [ ] Load test vector DB
- [ ] Implement rate limiting
- [ ] Add authentication/authorization
- [ ] Set up CI/CD pipeline
- [ ] Document API contracts
- [ ] Create runbooks for operations

---

## 📝 API Reference

### Document Ingestion
```python
from src.document_processor import document_processor

doc = document_processor.process_document("https://example.com/doc.pdf")
# Returns: {
#   "url": "...",
#   "title": "...",
#   "content": "...",
#   "pages": 50,
#   "metadata": {...}
# }
```

### Chunking
```python
from src.chunker import chunker

chunks = chunker.chunk_document(document)
# Returns: List[Chunk]
# Each chunk has: content, page_number, section, metadata
```

### Vector Storage
```python
from src.vector_store import vector_store

# Store chunks
vector_store.store_chunks(chunks, "doc_123")

# Retrieve similar chunks
results = vector_store.retrieve_relevant_chunks(
    query="What is AI?",
    document_id="doc_123",
    top_k=5
)
```

### RAG Query
```python
from src.langgraph_workflow import rag_workflow

result = rag_workflow.process_query(
    query="What are the main themes?",
    document_id="doc_123"
)
# Returns: {
#   "answer": "...",
#   "citations": [...],
#   "execution_time_seconds": 3.45
# }
```
