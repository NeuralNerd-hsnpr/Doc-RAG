"""
langgraph_workflow.py - LangGraph-based RAG workflow
Complete pipeline: Query → Routing → Retrieval → Synthesis → Answer
"""

import logging
import json
import traceback
from typing import Annotated, Dict, List, Literal, Optional
from dataclasses import dataclass, field
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from config import config
from src.vector_store import vector_store
from src.hf_llm import HuggingFaceLLM

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
    3. Synthesis: Generate answer with Hugging Face LLM
    4. Formatting: Add citations and structure
    """
    
    def __init__(self):
        """Initialize workflow with LangGraph"""
        logger.info("=" * 70)
        logger.info("Initializing RAG Workflow")
        logger.info("=" * 70)
        try:
            logger.info(f"Initializing Hugging Face LLM with model: {config.HF_MODEL}")
            logger.debug(f"HF_API_TOKEN available: {bool(config.HF_API_TOKEN)}")
            logger.debug(f"HF_MODEL: {config.HF_MODEL}")
            self.llm = HuggingFaceLLM()
            logger.info("✓ Hugging Face LLM initialized successfully")
        except Exception as e:
            logger.error("=" * 70)
            logger.error("Failed to initialize Hugging Face LLM")
            logger.error("=" * 70)
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            raise
        
        try:
            self.graph = self.build_graph()
            logger.info("✓ LangGraph workflow built successfully")
        except Exception as e:
            logger.error("Failed to build LangGraph workflow")
            logger.error(f"Error: {e}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            raise
        
        logger.info("=" * 70)
        logger.info("RAG Workflow Initialized Successfully")
        logger.info("=" * 70)
    
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
        logger.info("=" * 70)
        logger.info("ROUTER NODE: Starting Query Classification")
        logger.info("=" * 70)
        query = state["query"]
        logger.info(f"Query: {query[:200]}...")
        logger.info(f"Query length: {len(query)} characters")
        
        try:
            prompt = f"""Classify this query into one of these categories:
- search: general information search
- comparison: comparing different concepts
- summary: asking for summary/overview
- extraction: extracting specific facts
- reasoning: complex reasoning question

Query: {query}

Respond with only the category name."""
            
            logger.info("Calling LLM for query classification...")
            logger.debug(f"Router prompt length: {len(prompt)} characters")
            
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=50,
                temperature=0.1
            )
            
            logger.info(f"LLM response received: {response}")
            
            router_decision = response.strip().lower()
            logger.debug(f"Raw router decision: '{router_decision}'")
            
            valid_categories = ["search", "comparison", "summary", "extraction", "reasoning"]
            if router_decision not in valid_categories:
                logger.warning(f"Router decision '{router_decision}' not in valid categories, attempting to match...")
                for cat in valid_categories:
                    if cat in router_decision:
                        router_decision = cat
                        logger.info(f"Matched category: {cat}")
                        break
                else:
                    router_decision = "search"
                    logger.warning(f"Could not match category, defaulting to 'search'")
            
            state["router_decision"] = router_decision
            logger.info("=" * 70)
            logger.info(f"ROUTER NODE: Decision = {router_decision}")
            logger.info("=" * 70)
            return state
            
        except Exception as e:
            error_str = str(e).lower()
            error_msg_full = str(e)
            
            if "401" in error_str or "unauthorized" in error_str or "authentication" in error_str or "invalid" in error_str:
                error_summary = "Authentication failed with Hugging Face API. Please check your HF_API_TOKEN."
                error_msg = (
                    f"{error_summary}\n"
                    f"Get your API token from: https://huggingface.co/settings/tokens\n"
                    f"Make sure your .env file has: HF_API_TOKEN=hf_..."
                )
                logger.error(error_summary)
                logger.debug(f"Full error details: {error_msg_full}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                state["error"] = error_msg
                state["router_decision"] = "search"
                return state
            elif "503" in error_str or "loading" in error_str:
                error_summary = "Hugging Face model is loading. Please wait and try again."
                error_msg = (
                    f"{error_summary}\n"
                    f"The model needs 30-60 seconds to warm up on first request.\n"
                    f"Model: {config.HF_MODEL}"
                )
                logger.warning(error_summary)
                state["error"] = error_msg
                state["router_decision"] = "search"
                return state
            elif "429" in error_str or "rate limit" in error_str:
                error_summary = "Rate limit exceeded on Hugging Face API."
                error_msg = (
                    f"{error_summary}\n"
                    f"Free tier: 30,000 requests/month\n"
                    f"Wait a bit or upgrade your plan."
                )
                logger.error(error_summary)
                state["error"] = error_msg
                state["router_decision"] = "search"
                return state
            else:
                error_msg = f"Router error: {error_msg_full}"
                logger.error(error_msg)
                logger.debug(f"Router error details: {traceback.format_exc()}")
                state["error"] = error_msg
                state["router_decision"] = "search"
                return state
    
    def retrieval_node(self, state: RAGState) -> RAGState:
        """
        Node 2: Retrieval
        Get relevant chunks from Pinecone based on query
        """
        logger.info("=" * 70)
        logger.info("RETRIEVAL NODE: Starting Chunk Retrieval")
        logger.info("=" * 70)
        
        try:
            query = state["query"]
            document_id = state.get("document_id")
            router_decision = state.get("router_decision", "search")
            
            logger.info(f"Query: {query[:200]}...")
            logger.info(f"Document ID: {document_id or 'All documents'}")
            logger.info(f"Router decision: {router_decision}")
            
            top_k = config.RETRIEVAL_TOP_K
            logger.info(f"Base top_k: {top_k}")
            
            if router_decision == "comparison":
                top_k = config.RETRIEVAL_TOP_K + 2
                logger.info(f"Comparison query detected, increasing top_k to {top_k}")
            elif router_decision == "summary":
                top_k = config.RETRIEVAL_TOP_K + 3
                logger.info(f"Summary query detected, increasing top_k to {top_k}")
            
            logger.info("Calling vector_store.retrieve_relevant_chunks()...")
            retrieved = vector_store.retrieve_relevant_chunks(
                query=query,
                document_id=document_id,
                top_k=top_k
            )
            
            state["retrieved_chunks"] = retrieved
            
            logger.info("=" * 70)
            logger.info(f"RETRIEVAL NODE: Retrieved {len(retrieved)} chunks")
            logger.info("=" * 70)
            
            if retrieved:
                logger.debug(f"First chunk similarity: {retrieved[0].get('similarity', 'N/A')}")
                logger.debug(f"Last chunk similarity: {retrieved[-1].get('similarity', 'N/A')}")
            else:
                logger.warning("No chunks retrieved from vector store")
            
            return state
            
        except Exception as e:
            logger.error("=" * 70)
            logger.error("RETRIEVAL NODE: Error Occurred")
            logger.error("=" * 70)
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            state["error"] = f"Retrieval error: {e}"
            state["retrieved_chunks"] = []
            return state
    
    def synthesis_node(self, state: RAGState) -> RAGState:
        """
        Node 3: Synthesis
        Generate answer from retrieved chunks using Hugging Face LLM
        """
        logger.info("=" * 70)
        logger.info("SYNTHESIS NODE: Starting Answer Generation")
        logger.info("=" * 70)
        
        try:
            query = state["query"]
            chunks = state.get("retrieved_chunks", [])
            router_decision = state.get("router_decision", "search")
            
            logger.info(f"Query: {query[:200]}...")
            logger.info(f"Router decision: {router_decision}")
            logger.info(f"Number of chunks: {len(chunks)}")
            
            if not chunks:
                logger.warning("No chunks available for synthesis")
                state["synthesis"] = (
                    "No relevant information found in the document. "
                    "Try rephrasing your question or checking if the document "
                    "contains information about your topic."
                )
                state["citations"] = []
                logger.info("SYNTHESIS NODE: Completed (no chunks)")
                return state
            
            logger.info("Building context from retrieved chunks...")
            context = "RETRIEVED DOCUMENT SECTIONS:\n\n"
            for i, chunk in enumerate(chunks):
                metadata = chunk.get("metadata", {})
                similarity = chunk.get("similarity", 0)
                context += f"[SECTION {i+1}]\n"
                context += f"Page: {metadata.get('page_number', 'N/A')}\n"
                context += f"Topic: {metadata.get('section', 'N/A')}\n"
                context += f"Content Preview: {metadata.get('content_preview', '')}\n"
                context += f"Similarity: {similarity:.2f}\n\n"
                logger.debug(f"Chunk {i+1}: Page {metadata.get('page_number', 'N/A')}, "
                           f"Similarity: {similarity:.3f}")
            
            logger.info(f"Context built: {len(context)} characters")
            
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
            
            logger.info("Preparing LLM synthesis request...")
            logger.info(f"Model: {config.HF_MODEL}")
            logger.info(f"Prompt length: {len(prompt)} characters")
            logger.info(f"Max tokens: {config.MAX_TOKENS_RESPONSE}")
            logger.info(f"Temperature: {config.TEMPERATURE}")
            
            try:
                logger.info("Calling LLM.generate() for synthesis...")
                synthesis = self.llm.generate(
                    prompt=prompt,
                    max_tokens=config.MAX_TOKENS_RESPONSE,
                    temperature=config.TEMPERATURE,
                    system_prompt="You are a document analysis expert. Answer questions accurately based on the provided document sections."
                )
                
                state["synthesis"] = synthesis
                
                logger.info("=" * 70)
                logger.info("SYNTHESIS NODE: Completed Successfully")
                logger.info("=" * 70)
                logger.info(f"Response length: {len(synthesis)} characters")
                logger.debug(f"Response preview: {synthesis[:200]}...")
                return state
            except Exception as api_exception:
                error_str = str(api_exception).lower()
                error_msg_full = str(api_exception)
                
                if "401" in error_str or "unauthorized" in error_str or "authentication" in error_str or "invalid" in error_str:
                    error_summary = "Authentication failed with Hugging Face API. Please check your HF_API_TOKEN."
                    error_msg = (
                        f"{error_summary}\n"
                        f"Get your API token from: https://huggingface.co/settings/tokens\n"
                        f"Make sure your .env file has: HF_API_TOKEN=hf_..."
                    )
                    logger.error(error_summary)
                    logger.debug(f"Full error details: {error_msg_full}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                    state["error"] = error_msg
                    state["synthesis"] = (
                        "Unable to generate answer due to authentication error. "
                        "Please check your Hugging Face API token configuration in the .env file."
                    )
                    return state
                elif "503" in error_str or "loading" in error_str:
                    error_summary = "Hugging Face model is loading. Please wait and try again."
                    error_msg = (
                        f"{error_summary}\n"
                        f"The model needs 30-60 seconds to warm up on first request.\n"
                        f"Model: {config.HF_MODEL}"
                    )
                    logger.warning(error_summary)
                    state["error"] = error_msg
                    state["synthesis"] = "Model is loading. Please wait 30-60 seconds and try again."
                    return state
                elif "429" in error_str or "rate limit" in error_str:
                    error_summary = "Rate limit exceeded on Hugging Face API."
                    error_msg = (
                        f"{error_summary}\n"
                        f"Free tier: 30,000 requests/month\n"
                        f"Wait a bit or upgrade your plan."
                    )
                    logger.error(error_summary)
                    state["error"] = error_msg
                    state["synthesis"] = "Rate limit exceeded. Please wait or upgrade your plan."
                    return state
                else:
                    error_msg = f"Error in synthesis: {error_msg_full}"
                    logger.error(error_msg)
                    logger.debug(f"Exception details: {traceback.format_exc()}")
                    state["error"] = error_msg
                    state["synthesis"] = "Error generating answer. Please try again."
                    return state
            
        except Exception as e:
            error_msg = f"Synthesis error: {e}"
            logger.error(error_msg)
            logger.debug(f"Exception details: {traceback.format_exc()}")
            state["error"] = error_msg
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
        logger.info("=" * 70)
        logger.info("RAG WORKFLOW: Starting Query Processing")
        logger.info("=" * 70)
        logger.info(f"Query: {query}")
        logger.info(f"Query length: {len(query)} characters")
        logger.info(f"Document ID: {document_id or 'All documents'}")
        
        start_time = datetime.now()
        logger.debug(f"Start time: {start_time.isoformat()}")
        
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
        
        logger.info("Invoking LangGraph workflow...")
        try:
            final_state = self.graph.invoke(initial_state)
            logger.info("LangGraph workflow completed")
        except Exception as e:
            logger.error("=" * 70)
            logger.error("RAG WORKFLOW: Error During Execution")
            logger.error("=" * 70)
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            raise
        
        execution_time = (datetime.now() - start_time).total_seconds()
        final_state["execution_time"] = execution_time
        
        logger.info("=" * 70)
        logger.info("RAG WORKFLOW: Query Processing Completed")
        logger.info("=" * 70)
        logger.info(f"Execution time: {execution_time:.2f} seconds")
        logger.info(f"Router decision: {final_state.get('router_decision')}")
        logger.info(f"Chunks retrieved: {len(final_state.get('retrieved_chunks', []))}")
        logger.info(f"Answer length: {len(final_state.get('synthesis', ''))} characters")
        logger.info(f"Citations: {len(final_state.get('citations', []))}")
        if final_state.get("error"):
            logger.warning(f"Error in workflow: {final_state.get('error')}")
        
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