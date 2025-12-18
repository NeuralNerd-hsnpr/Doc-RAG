import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from pinecone import Pinecone
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_index_dimension():
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    
    try:
        index_info = pc.describe_index(config.PINECONE_INDEX_NAME)
        dimension = index_info.dimension
        
        index = pc.Index(config.PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        vector_count = stats.total_vector_count
        
        logger.info(f"Index: {config.PINECONE_INDEX_NAME}")
        logger.info(f"Current dimension: {dimension}")
        logger.info(f"Total vectors: {vector_count}")
        
        from src.embeddings import embedding_generator
        expected_dimension = embedding_generator.dimension
        
        logger.info(f"Expected dimension: {expected_dimension}")
        
        if dimension != expected_dimension:
            logger.error(f"MISMATCH: Index dimension ({dimension}) != Embedding dimension ({expected_dimension})")
            return False
        else:
            logger.info("Dimensions match!")
            return True
            
    except Exception as e:
        logger.error(f"Error checking index: {e}")
        return False


def delete_index():
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    
    try:
        logger.warning(f"Deleting index: {config.PINECONE_INDEX_NAME}")
        pc.delete_index(config.PINECONE_INDEX_NAME)
        logger.info("Index deleted successfully")
        return True
    except Exception as e:
        logger.error(f"Error deleting index: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check or fix Pinecone index dimension")
    parser.add_argument("--check", action="store_true", help="Check index dimension")
    parser.add_argument("--delete", action="store_true", help="Delete existing index")
    
    args = parser.parse_args()
    
    if args.delete:
        confirm = input(f"Are you sure you want to delete index '{config.PINECONE_INDEX_NAME}'? (yes/no): ")
        if confirm.lower() == "yes":
            delete_index()
        else:
            logger.info("Cancelled")
    elif args.check:
        check_index_dimension()
    else:
        parser.print_help()

