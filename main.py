"""
main.py - Interactive CLI for Document RAG System
Handles document ingestion, vectorization, and Q&A
"""

import logging
import sys
from typing import Optional
import json
from datetime import datetime

from config import config
from src.document_processor import document_processor
from src.chunker import chunker
from src.vector_store import vector_store
from src.langgraph_workflow import rag_workflow

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class RAGCLISystem:
    """
    Interactive CLI for Document RAG System
    Main entry point for users
    """
    
    def __init__(self):
        """Initialize CLI system"""
        self.current_document_id = None
        self.document_metadata = {}
        logger.info("RAG CLI System initialized")
    
    def print_header(self):
        """Print system header"""
        print("\n" + "="*70)
        print("  DOCUMENT RAG SYSTEM - Interactive Q&A")
        print("="*70 + "\n")
    
    def print_menu(self):
        """Print main menu"""
        print("\nMAIN MENU")
        print("-" * 70)
        print("1. Ingest Document (from URL)")
        print("2. Ask Question (about current document)")
        print("3. Show Document Info")
        print("4. Show Vector Store Stats")
        print("5. Delete Document from Index")
        print("6. Exit")
        print("-" * 70)
    
    def ingest_document(self):
        """
        Step 1: Ingest document from URL
        Download → Extract → Chunk → Vectorize → Store
        """
        print("\n[STEP 1: DOCUMENT INGESTION]")
        print("-" * 70)
        
        # Get URL from user
        url = input("Enter document URL (or press Enter to skip): ").strip()
        if not url:
            print("Skipped document ingestion")
            return False
        
        print(f"\nProcessing: {url}")
        
        # Step 1: Process document
        print("\n→ Downloading and extracting PDF...")
        document = document_processor.process_document(url)
        if not document:
            print("✗ Failed to process document")
            return False
        
        print(f"✓ Document processed:")
        print(f"  Title: {document['title']}")
        print(f"  Pages: {document['pages']}")
        print(f"  Content length: {len(document['content'])} characters")
        
        # Step 2: Chunk document
        print("\n→ Chunking document...")
        chunks = chunker.chunk_document(document)
        print(f"✓ Created {len(chunks)} chunks")
        
        # Show chunk sample
        if chunks:
            sample_chunk = chunks[0]
            print(f"\nSample chunk (Page {sample_chunk.page_number}):")
            print(f"  Section: {sample_chunk.section}")
            print(f"  Content preview: {sample_chunk.content[:150]}...")
            print(f"  Tokens: {sample_chunk.metadata.get('token_count', 'N/A')}")
        
        # Step 3: Vectorize and store
        print("\n→ Generating embeddings and storing in Pinecone...")
        document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        success = vector_store.store_chunks(chunks, document_id)
        if not success:
            print("✗ Failed to store vectors")
            return False
        
        print(f"✓ Vectors stored successfully")
        print(f"  Document ID: {document_id}")
        
        # Update CLI state
        self.current_document_id = document_id
        self.document_metadata = {
            "url": url,
            "title": document['title'],
            "pages": document['pages'],
            "chunks": len(chunks),
            "ingested_at": datetime.now().isoformat()
        }
        
        # Show next steps
        print("\n" + "-"*70)
        print(f"✓ Document successfully ingested!")
        print(f"  You can now ask questions about this document")
        print(f"  Type 'help' for available commands")
        
        return True
    
    def ask_question(self):
        """
        Step 2: Ask question about document
        Query → Retrieve → Synthesize → Answer with Citations
        """
        if not self.current_document_id:
            print("\n✗ No document loaded. Please ingest a document first.")
            return
        
        print("\n[STEP 2: Q&A]")
        print("-" * 70)
        print(f"Document: {self.document_metadata.get('title', 'Unknown')}")
        print(f"Document ID: {self.current_document_id}")
        print("-" * 70)
        
        # Get question from user
        query = input("\nEnter your question (or 'back' to return): ").strip()
        if query.lower() == 'back':
            return
        
        if not query:
            print("Please enter a valid question")
            return
        
        print(f"\nProcessing: {query}")
        print("\nGenerating answer...")
        
        # Process query through RAG pipeline
        result = rag_workflow.process_query(
            query=query,
            document_id=self.current_document_id
        )
        
        # Display results
        print("\n" + "="*70)
        print("ANSWER")
        print("="*70)
        print(f"\n{result['answer']}\n")
        
        # Display citations
        if result['citations']:
            print("="*70)
            print("CITATIONS")
            print("="*70)
            for citation in result['citations']:
                print(f"\n[SECTION {citation['section']}]")
                print(f"  Page: {citation['page']}")
                print(f"  Topic: {citation['topic']}")
                print(f"  Relevance Score: {citation['similarity_score']}")
        
        # Display metadata
        print("\n" + "="*70)
        print("METADATA")
        print("="*70)
        print(f"Query Type: {result['router_decision']}")
        print(f"Chunks Retrieved: {result['chunks_retrieved']}")
        print(f"Processing Time: {result['execution_time_seconds']:.2f}s")
        
        if result['error']:
            print(f"⚠ Warning: {result['error']}")
        
        # Option to save result
        save = input("\nSave result to file? (y/n): ").lower()
        if save == 'y':
            self.save_result(result)
    
    def show_document_info(self):
        """Show information about current document"""
        if not self.current_document_id:
            print("\n✗ No document loaded")
            return
        
        print("\n[DOCUMENT INFORMATION]")
        print("-" * 70)
        for key, value in self.document_metadata.items():
            print(f"{key:.<20} {value}")
        
        # Show vector store stats
        stats = vector_store.get_index_stats()
        print("\n[VECTOR STORE STATS]")
        print(f"Total vectors: {stats.get('total_vector_count', 'N/A')}")
        print(f"Dimension: {stats.get('dimension', 'N/A')}")
    
    def show_vector_store_stats(self):
        """Show vector store statistics"""
        print("\n[VECTOR STORE STATISTICS]")
        print("-" * 70)
        
        stats = vector_store.get_index_stats()
        
        print(f"Index Name: {vector_store.index_name}")
        print(f"Total Vectors: {stats.get('total_vector_count', 'N/A')}")
        print(f"Dimension: {stats.get('dimension', 'N/A')}")
        
        if stats.get('namespaces'):
            print(f"\nNamespaces: {len(stats['namespaces'])}")
            for ns, info in stats['namespaces'].items():
                print(f"  {ns}: {info.get('vector_count', 0)} vectors")
    
    def delete_document(self):
        """Delete document from vector store"""
        if not self.current_document_id:
            print("\n✗ No document loaded")
            return
        
        confirm = input(
            f"\nDelete document '{self.document_metadata.get('title')}'? (y/n): "
        ).lower()
        
        if confirm != 'y':
            print("Cancelled")
            return
        
        print("Deleting from vector store...")
        success = vector_store.delete_document_vectors(self.current_document_id)
        
        if success:
            print("✓ Document deleted")
            self.current_document_id = None
            self.document_metadata = {}
        else:
            print("✗ Failed to delete document")
    
    def save_result(self, result: dict):
        """Save Q&A result to file"""
        filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✓ Result saved to: {filename}")
    
    def run(self):
        """Main CLI loop"""
        self.print_header()
        
        print("Welcome to the Document RAG System!")
        print("\nThis system allows you to:")
        print("1. Upload documents from URLs")
        print("2. Automatically chunk and vectorize them")
        print("3. Ask questions and get precise answers with citations")
        
        while True:
            try:
                self.print_menu()
                choice = input("Select option (1-6): ").strip()
                
                if choice == '1':
                    self.ingest_document()
                elif choice == '2':
                    self.ask_question()
                elif choice == '3':
                    self.show_document_info()
                elif choice == '4':
                    self.show_vector_store_stats()
                elif choice == '5':
                    self.delete_document()
                elif choice == '6':
                    print("\n✓ Exiting RAG System. Goodbye!")
                    break
                else:
                    print("Invalid option. Please try again.")
            
            except KeyboardInterrupt:
                print("\n\n✓ Exiting RAG System. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in CLI: {e}")
                print(f"\n✗ Error: {e}")
                print("Please try again")


def main():
    """Entry point for the application"""
    try:
        # Verify configuration
        config.validate()
        
        # Run CLI
        cli = RAGCLISystem()
        cli.run()
        
    except ValueError as e:
        print(f"\n✗ Configuration Error: {e}")
        print("\nPlease:")
        print("1. Copy .env.example to .env")
        print("2. Fill in your Anthropic and Pinecone credentials")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n✗ Fatal Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()