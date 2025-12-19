import sys
import logging
from config import config
from src.vector_store import vector_store
from src.query_processor import query_processor

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_retrieval(query: str, document_id: str = None):
    print(f"\n{'='*70}")
    print(f"Testing Retrieval for Query: '{query}'")
    print(f"{'='*70}\n")
    
    processed_query = query_processor.preprocess_query(query)
    print(f"Processed query: '{processed_query}'\n")
    
    chunks = vector_store.retrieve_relevant_chunks(
        query=query,
        document_id=document_id,
        top_k=10
    )
    
    print(f"\n{'='*70}")
    print(f"Retrieval Results: {len(chunks)} chunks found")
    print(f"{'='*70}\n")
    
    if chunks:
        for i, chunk in enumerate(chunks[:5], 1):
            print(f"Chunk {i}:")
            print(f"  Similarity: {chunk['similarity']:.4f}")
            print(f"  Page: {chunk['page_number']}")
            print(f"  Section: {chunk['section']}")
            print(f"  Content preview: {chunk['content'][:200]}...")
            print()
    else:
        print("No chunks retrieved!")
        print("\nTroubleshooting:")
        print("1. Check if document was ingested")
        print("2. Check similarity threshold (current: {})".format(config.SIMILARITY_THRESHOLD))
        print("3. Try a more specific query")
    
    return chunks


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_retrieval.py <query> [document_id]")
        print("\nExample:")
        print("  python test_retrieval.py 'what is the topic'")
        print("  python test_retrieval.py 'what is the topic' doc_20240115_103000")
        sys.exit(1)
    
    query = sys.argv[1]
    document_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    test_retrieval(query, document_id)

