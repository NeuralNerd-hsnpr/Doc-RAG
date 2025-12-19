# Fixing Pinecone Index Dimension Mismatch

## Problem

If you see an error like:
```
Vector dimension 384 does not match the dimension of the index 1536
```

This means your existing Pinecone index was created with a different dimension than your current embedding model produces.

## Solution Options

### Option 1: Delete and Recreate Index (Recommended if index is empty or can be recreated)

```bash
python scripts/fix_index_dimension.py --delete
```

Then run your application again - it will create a new index with the correct dimension.

### Option 2: Use a Different Index Name

Add to your `.env` file:
```bash
PINECONE_INDEX_NAME=document-rag-index-384
```

This creates a new index with the correct dimension while keeping the old one.

### Option 3: Change Embedding Model to Match Existing Index

If your index has important data, change the embedding model to match:

Add to your `.env` file:
```bash
EMBEDDING_MODEL=intfloat/e5-large-v2
```

This model produces 1024-dimensional embeddings. For 1536 dimensions, you would need a different model or use OpenAI embeddings.

## Check Current Status

To check your index dimension:
```bash
python scripts/fix_index_dimension.py --check
```

## Current Configuration

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding Dimension**: 384
- **Index Name**: `document-rag-index` (configurable via `PINECONE_INDEX_NAME`)

## Supported Embedding Models and Dimensions

| Model | Dimension |
|-------|-----------|
| sentence-transformers/all-MiniLM-L6-v2 | 384 |
| sentence-transformers/all-mpnet-base-v2 | 768 |
| intfloat/e5-large-v2 | 1024 |
| BAAI/bge-large-en-v1.5 | 1024 |
| BAAI/bge-base-en-v1.5 | 768 |

Set `EMBEDDING_MODEL` in `.env` to change the model.

