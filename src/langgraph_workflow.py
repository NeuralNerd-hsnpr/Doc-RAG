"""
langgraph_workflow.py - LangGraph-based RAG workflow
Complete pipeline: Query → Routing → Retrieval → Synthesis → Answer
"""

import logging
import json
from typing import Annotated, Dict, List, Literal, Optional
from dataclasses import dataclass, field
from datetime import datetime

import anthropic
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from config import config
from src.vector_store import vector_store

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """State passed through LangGraph workflow"""
    query: str
    document_id: Optional[str]
    router_decision: Optional[str]
    retrieved_chunks: List[Dict]
    synthesis: str
    citations: List[Dict]
    execution_time: float
    error: Optional[str]


class RAGWorkflow:
    """
    Complete RAG workflow using LangGraph:
    1. Router: Classify query intent
    2. Retrieval: Get relevant chunks from Pinecone
    3. Synthesis: Generate answer with Claude
    4. Formatting: Add citations and structure
    """
    
    def __init__(self):
        """Initialize workflow with LangGraph"""
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.graph = self.build_graph()
        
        logger.info("RAG workflow initialized")
    
    def build_graph(self):
        """Build LangGraph workflow"""
        graph = StateGraph(RAGState)
        
        # Add nodes
        graph.add_node("router", self.router_node)
        graph.add_node("retrieval", self.retrieval_node)
        graph.add_node("synthesis", self.synthesis_node)
        graph.add_node("formatter", self.formatter_node)
        
        # Add edges
        graph.add_edge(START, "router")
        graph.add_edge("router", "retrieval")
        graph.add_edge("retrieval", "synthesis")
        graph.add_edge("synthesis", "formatter")
        graph.add_edge("formatter", END)
        
        return graph.compile()
    
    def router_node(self, state: RAGState) -> RAGState:
        """
        Node 1: Router
        Classify query intent and determine search strategy
        """
        logger.info(f"Router: Processing query - {state['query'][:100]}...")
        
        try:
            query = state["query"]
            
            # Use Claude to classify query
            response = self.client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": f"""Classify this query:
- search: general information search
- comparison: comparing different concepts
- summary: asking for summary/overview
- extraction: extracting specific facts
- reasoning: complex reasoning question

Query: {query}

Respond with only the category."""
                }]
            )
            
            router_decision = response.content[0].text.strip().lower()
            state["router_decision"] = router_decision
            
            logger.info(f"Router decision: {router_decision}")
            return state
            
        except Exception as e:
            logger.error(f"Router error: {e}")
            state["error"] = f"Router error: {e}"
            return state
    
    def retrieval_node(self, state: RAGState) -> RAGState:
        """
        Node 2: Retrieval
        Get relevant chunks from Pinecone based on query
        """
        logger.info(f"Retrieval: Getting relevant chunks...")
        
        try:
            query = state["query"]
            document_id = state.get("document_id")
            
            # Determine how many chunks to retrieve based on query type
            router_decision = state.get("router_decision", "search")
            top_k = config.RETRIEVAL_TOP_K
            
            if router_decision == "comparison":
                top_k = config.RETRIEVAL_TOP_K + 2
            elif router_decision == "summary":
                top_k = config.RETRIEVAL_TOP_K + 3
            
            # Retrieve from Pinecone
            retrieved = vector_store.retrieve_relevant_chunks(
                query=query,
                document_id=document_id,
                top_k=top_k
            )
            
            state["retrieved_chunks"] = retrieved
            
            logger.info(f"Retrieved {len(retrieved)} chunks")
            return state
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            state["error"] = f"Retrieval error: {e}"
            state["retrieved_chunks"] = []
            return state
    
    def synthesis_node(self, state: RAGState) -> RAGState:
        """
        Node 3: Synthesis
        Generate answer from retrieved chunks using Claude
        """
        logger.info(f"Synthesis: Generating answer...")
        
        try:
            query = state["query"]
            chunks = state.get("retrieved_chunks", [])
            
            if not chunks:
                state["synthesis"] = (
                    "No relevant information found in the document. "
                    "Try rephrasing your question or checking if the document "
                    "contains information about your topic."
                )
                state["citations"] = []
                return state
            
            # Build context from retrieved chunks
            context = "RETRIEVED DOCUMENT SECTIONS:\n\n"
            for i, chunk in enumerate(chunks):
                metadata = chunk.get("metadata", {})
                context += f"[SECTION {i+1}]\n"
                context += f"Page: {metadata.get('page_number', 'N/A')}\n"
                context += f"Topic: {metadata.get('section', 'N/A')}\n"
                context += f"Content Preview: {metadata.get('content_preview', '')}\n"
                context += f"Similarity: {chunk.get('similarity', 0):.2f}\n\n"
            
            # Create synthesis prompt
            prompt = f"""You are a document analysis expert. Using ONLY the provided document sections, 
answer the following question accurately and precisely.

IMPORTANT RULES:
1. Answer ONLY using information from the provided sections
2. If information is not in the document, say so clearly
3. Cite which section you're using (e.g., [SECTION 1])
4. Be concise but thorough
5. Organize your answer clearly

Question: {query}

{context}

Answer:"""
            
            # Generate answer
            response = self.client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=config.MAX_TOKENS_RESPONSE,
                temperature=config.TEMPERATURE,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            synthesis = response.content[0].text
            state["synthesis"] = synthesis
            
            logger.info(f"Synthesis completed")
            return state
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            state["error"] = f"Synthesis error: {e}"
            state["synthesis"] = "Error generating answer. Please try again."
            return state
    
    def formatter_node(self, state: RAGState) -> RAGState:
        """
        Node 4: Formatter
        Extract and format citations from synthesis
        """
        logger.info(f"Formatter: Processing citations...")
        
        try:
            synthesis = state["synthesis"]
            chunks = state.get("retrieved_chunks", [])
            
            # Extract citations from answer
            import re
            citations = []
            
            # Find all [SECTION N] references
            matches = re.findall(r'\[SECTION (\d+)\]', synthesis)
            
            for section_num in set(matches):
                try:
                    idx = int(section_num) - 1
                    if idx < len(chunks):
                        chunk = chunks[idx]
                        metadata = chunk.get("metadata", {})
                        
                        citation = {
                            "section": section_num,
                            "page": metadata.get("page_number", "N/A"),
                            "topic": metadata.get("section", "N/A"),
                            "similarity_score": round(chunk.get("similarity", 0), 3),
                            "source_document": metadata.get("source", "Unknown")
                        }
                        citations.append(citation)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error processing citation: {e}")
            
            state["citations"] = citations
            logger.info(f"Extracted {len(citations)} citations")
            
            return state
            
        except Exception as e:
            logger.error(f"Formatter error: {e}")
            state["citations"] = []
            return state
    
    def process_query(
        self, 
        query: str, 
        document_id: Optional[str] = None
    ) -> Dict:
        """
        Main entry point: Process query through entire pipeline
        
        Args:
            query: User question
            document_id: Optional specific document to search
            
        Returns:
            Complete response with answer and citations
        """
        logger.info(f"Processing query: {query}")
        
        start_time = datetime.now()
        
        # Initialize state
        initial_state: RAGState = {
            "query": query,
            "document_id": document_id,
            "router_decision": None,
            "retrieved_chunks": [],
            "synthesis": "",
            "citations": [],
            "execution_time": 0.0,
            "error": None
        }
        
        # Run workflow
        final_state = self.graph.invoke(initial_state)
        
        # Calculate execution time
        final_state["execution_time"] = (
            datetime.now() - start_time
        ).total_seconds()
        
        return {
            "question": query,
            "answer": final_state["synthesis"],
            "citations": final_state["citations"],
            "router_decision": final_state.get("router_decision"),
            "chunks_retrieved": len(final_state.get("retrieved_chunks", [])),
            "execution_time_seconds": final_state["execution_time"],
            "error": final_state.get("error")
        }


# Global workflow instance
rag_workflow = RAGWorkflow()