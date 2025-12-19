"""
langgraph_workflow.py - LangGraph-based RAG workflow
Complete pipeline: Query → Routing → Retrieval → Synthesis → Answer
"""

import logging
import json
import traceback
import uuid
import time
from typing import Annotated, Dict, List, Literal, Optional
from dataclasses import dataclass, field
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from config import config
from src.vector_store import vector_store
from src.hf_llm import HuggingFaceLLM
from src.logger import request_logger

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """State passed through LangGraph workflow"""
    query: str
    document_id: Optional[str]
    request_id: Optional[str]
    router_decision: Optional[str]
    retrieved_chunks: List[Dict]
    synthesis: str
    citations: List[Dict]
    execution_time: float
    error: Optional[str]
    is_retrospective: Optional[bool]


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
        request_id = state.get("request_id", str(uuid.uuid4()))
        state["request_id"] = request_id
        
        query = state["query"]
        document_id = state.get("document_id")
        
        request_logger.log_request_start(request_id, query, document_id)
        
        start_time = time.time()
        
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
            
            router_time = (time.time() - start_time) * 1000
            request_logger.log_router_decision(request_id, router_decision)
            
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
        request_id = state.get("request_id", str(uuid.uuid4()))
        start_time = time.time()
        
        try:
            query = state["query"]
            document_id = state.get("document_id")
            router_decision = state.get("router_decision", "search")
            
            query_lower = query.lower()
            is_general_question = any(word in query_lower for word in [
                "topic", "theme", "subject", "about", "main", "overview", 
                "summary", "what is", "discuss", "cover", "include"
            ])
            is_retrospective = any(word in query_lower for word in [
                "played out", "happened", "occurred", "was", "were", 
                "did", "past", "previous", "retrospective", "review"
            ])
            
            top_k = config.RETRIEVAL_TOP_K
            if is_general_question:
                top_k = max(config.RETRIEVAL_TOP_K * 2, 20)
                logger.info(f"[RETRIEVAL] General question detected, increasing top_k to {top_k}")
            elif router_decision == "comparison":
                top_k = config.RETRIEVAL_TOP_K + 5
            elif router_decision == "summary":
                top_k = config.RETRIEVAL_TOP_K + 8
            
            if is_retrospective:
                logger.info(f"[RETRIEVAL] Retrospective question detected: '{query}'")
                state["is_retrospective"] = True
            
            retrieved = vector_store.retrieve_relevant_chunks(
                query=query,
                document_id=document_id,
                top_k=top_k
            )
            
            state["retrieved_chunks"] = retrieved
            
            retrieval_time = (time.time() - start_time) * 1000
            similarities = [c.get("similarity", 0) for c in retrieved]
            request_logger.log_retrieval(request_id, len(retrieved), similarities, retrieval_time)
            
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
        request_id = state.get("request_id", str(uuid.uuid4()))
        start_time = time.time()
        
        try:
            query = state["query"]
            chunks = state.get("retrieved_chunks", [])
            
            if not chunks:
                logger.warning(f"[SYNTHESIS] No chunks available for query: '{query}'")
                logger.warning("[SYNTHESIS] This might indicate:")
                logger.warning("  1. Similarity threshold too high")
                logger.warning("  2. Query doesn't match document content")
                logger.warning("  3. Index might be empty or wrong document_id")
                
                state["synthesis"] = (
                    "No relevant information found in the document. "
                    "Try rephrasing your question or checking if the document "
                    "contains information about your topic."
                )
                state["citations"] = []
                return state
            
            logger.info(f"[SYNTHESIS] Processing {len(chunks)} chunks for synthesis")
            
            logger.info("Building context from retrieved chunks...")
            
            chunks = self._deduplicate_chunks(chunks)
            
            context = "RETRIEVED DOCUMENT SECTIONS:\n\n"
            for i, chunk in enumerate(chunks):
                metadata = chunk.get("metadata", {})
                similarity = chunk.get("similarity", 0)
                content = chunk.get("content", metadata.get("content_preview", ""))
                
                if len(content) > 1000:
                    content = content[:1000] + "..."
                
                context += f"[SECTION {i+1}]\n"
                context += f"Page: {metadata.get('page_number', 'N/A')}\n"
                context += f"Section: {metadata.get('section', 'N/A')}\n"
                context += f"Relevance Score: {similarity:.3f}\n"
                context += f"Content:\n{content}\n\n"
                
                logger.debug(
                    f"Chunk {i+1}: Page {metadata.get('page_number', 'N/A')}, "
                    f"Section: {metadata.get('section', 'N/A')}, "
                    f"Similarity: {similarity:.3f}, "
                    f"Content length: {len(content)} chars"
                )
            
            logger.info(f"Context built: {len(context)} characters")
            
            system_prompt_text = """You are an expert document analyst. Your task is to provide clear, comprehensive, and accurate answers based ONLY on the provided document sections.

CRITICAL ANTI-REPETITION RULES:
1. NEVER repeat the same sentence, phrase, or information multiple times
2. NEVER use phrases like "The sections suggest" repeatedly - state facts directly
3. NEVER generate repetitive patterns like "X has Y% since [month]" with different months
4. If you find yourself writing similar sentences, STOP and consolidate into one clear statement
5. Each sentence must provide NEW, UNIQUE information

LOGIC AND ACCURACY RULES:
6. NUMBER RULE: Only quote exact numbers/statistics from the document. DO NOT invent, generate, or vary numbers. If the document says "3%", quote "3%" - do NOT create variations
7. TEMPORAL LOGIC: Check if the question asks about past events but the document is forward-looking. If so, clearly state this mismatch
8. NEGATIVE ANSWER RULE: If information is NOT in the retrieved sections, say "I could not find information about [topic] in the retrieved sections" - DO NOT claim "the document does not address" unless absolutely certain
9. Use [SECTION N] citations for each specific fact
10. Synthesize information from multiple sections into a coherent answer

ANSWER STRUCTURE:
1. Direct answer (1-2 sentences)
2. Supporting details with citations [SECTION N]
3. Comprehensive coverage for general/topic questions

QUALITY REQUIREMENTS:
- No repetition of sentences, phrases, or patterns
- Each sentence must provide unique information
- Maximum 300 words
- Quote exact numbers - do not generate variations
- Be specific and factual, avoid vague generalizations
- For topic questions, list ALL major topics/themes found in retrieved sections"""

            query_lower = query.lower()
            is_topic_question = any(word in query_lower for word in ["topic", "subject", "theme", "about", "main"])
            is_specific_question = any(word in query_lower for word in ["which", "what", "who", "when", "where", "how"])
            is_retrospective = state.get("is_retrospective", False) or any(word in query_lower for word in [
                "played out", "happened", "occurred", "was", "were", "did", "past"
            ])
            
            if is_retrospective:
                instruction = """CRITICAL TEMPORAL LOGIC CHECK:
This question asks about past events ("played out", "happened", "was", "were"). 
However, this document appears to be a forward-looking forecast/outlook document.

YOU MUST:
1. Check the document date/context - if it's a forecast document, state clearly:
   "This document contains forward-looking forecasts (published in [year]) rather than a review of past events."
2. List the forecasts mentioned in the document
3. DO NOT claim events "played out" if the document only forecasts them
4. DO NOT generate fake statistics or dates - only quote what's in the document
5. If the document mentions "so far this year" or similar, clarify this refers to partial year data, not full outcomes"""
            elif is_topic_question:
                instruction = "Provide a comprehensive summary of ALL main topics covered in the document. Include all major themes and sections mentioned."
            elif is_specific_question:
                instruction = "Answer the question directly with specific facts from the document. Be precise and cite sources."
            else:
                instruction = "Answer the question based on the document sections. Be clear and comprehensive."
            
            prompt = f"""Question: {query}

{instruction}

CRITICAL REQUIREMENTS:
1. Answer directly - do not use phrases like "The sections suggest" or "The document mentions" repeatedly
2. State facts directly: "The document discusses X" or "According to [SECTION N], Y"
3. Do NOT repeat the same sentence or information - each sentence must be unique
4. Maximum 300 words
5. Cite each fact with [SECTION N]
6. NEGATIVE ANSWER RULE: If you cannot find information in the RETRIEVED sections, say "I could not find information about [topic] in the retrieved sections" - DO NOT claim "the document does not address" or "the document does not contain" unless you are absolutely certain after reviewing ALL retrieved sections
7. NUMBER RULE: Only quote exact numbers/statistics from the document. DO NOT generate or invent numbers. If you see "3%" in one section, do NOT repeat it with different months or contexts
8. LOGIC CHECK: If you find yourself repeating similar phrases with slight variations (e.g., "X has Y% since [month]"), STOP and consolidate into one statement
9. Be comprehensive - if asking about topics/themes, mention ALL major topics found in the retrieved sections

Document Sections:
{context}

Answer:"""
            
            try:
                stop_sequences = [
                    "\n\n\n",
                    "---",
                    "===",
                    "The sections suggest",
                    "The document suggests"
                ]
                
                synthesis = self.llm.generate(
                    prompt=prompt,
                    max_tokens=min(config.MAX_TOKENS_RESPONSE, 600),
                    temperature=0.4,
                    system_prompt=system_prompt_text,
                    repetition_penalty=1.15,
                    stop_sequences=stop_sequences
                )
                
                synthesis = self._post_process_answer(synthesis, query, chunks)
                state["synthesis"] = synthesis
                
                synthesis_time = (time.time() - start_time) * 1000
                request_logger.log_synthesis(
                    request_id,
                    len(prompt),
                    len(synthesis),
                    synthesis_time,
                    config.HF_MODEL
                )
                
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
    
    def _post_process_answer(self, answer: str, query: str, chunks: List[Dict]) -> str:
        import re
        
        answer = answer.strip()
        
        if not answer:
            return "I could not generate an answer from the provided document sections."
        
        special_tokens = [
            r'\[/ASS\]', r'\[ASS\]', r'</s>', r'<s>', r'\[INST\]', r'\[/INST\]',
            r'<\|assistant\|>', r'<\|user\|>', r'<\|system\|>',
            r'### Assistant:', r'### User:', r'### System:',
            r'<|endoftext|>', r'<|end|>'
        ]
        
        for token_pattern in special_tokens:
            answer = re.sub(token_pattern, '', answer, flags=re.IGNORECASE)
        
        answer = re.sub(r'\s+', ' ', answer)
        
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        
        seen_sentences = set()
        unique_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
            
            sentence_lower = sentence.lower()
            
            if "the sections suggest" in sentence_lower and len(unique_sentences) > 0:
                continue
            
            is_duplicate = False
            for seen in seen_sentences:
                similarity = self._sentence_similarity(sentence_lower, seen.lower())
                if similarity > 0.80:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_sentences.append(sentence)
                seen_sentences.add(sentence_lower)
        
        if not unique_sentences:
            return answer
        
        processed = ' '.join(unique_sentences)
        
        processed = re.sub(r'\s+', ' ', processed)
        
        processed = self._detect_and_fix_repetition_loops(processed)
        
        processed = self._validate_numbers(processed, chunks)
        
        negative_claims = re.findall(
            r'the document (?:does not|doesn\'t|did not|didn\'t) (?:address|contain|discuss|mention|include)',
            processed.lower()
        )
        if negative_claims:
            logger.warning(f"[POST_PROCESS] Found {len(negative_claims)} negative claims - may be hallucination")
        
        if len(processed) > 1200:
            sentences = processed.split('. ')
            processed = '. '.join(sentences[:8]) + '.'
        
        processed = processed.strip()
        
        if processed != answer:
            logger.info(f"[POST_PROCESS] Removed {len(sentences) - len(unique_sentences)} duplicate/redundant sentences")
            logger.debug(f"[POST_PROCESS] Original length: {len(answer)}, Processed length: {len(processed)}")
        
        return processed
    
    def _detect_and_fix_repetition_loops(self, text: str) -> str:
        import re
        
        patterns = [
            r'([^.]{10,100})\s+\1\s+\1',
            r'(\w+\s+has\s+\w+\s+\d+%[^.]{0,50})\.\s+\1',
            r'(\w+\s+since\s+\w+[^.]{0,30})\.\s+\1',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                repeated_phrase = match.group(1)
                logger.warning(f"[POST_PROCESS] Detected repetition loop: '{repeated_phrase[:50]}...'")
                text = text.replace(match.group(0), repeated_phrase + '.', 1)
        
        sentences = text.split('. ')
        seen_patterns = {}
        cleaned_sentences = []
        
        for sentence in sentences:
            sentence_clean = re.sub(r'\d+%', '%', sentence.lower())
            sentence_clean = re.sub(r'\d+', 'N', sentence_clean)
            
            if sentence_clean in seen_patterns:
                if seen_patterns[sentence_clean] >= 2:
                    logger.debug(f"[POST_PROCESS] Skipping repetitive sentence pattern")
                    continue
                seen_patterns[sentence_clean] += 1
            else:
                seen_patterns[sentence_clean] = 1
            
            cleaned_sentences.append(sentence)
        
        return '. '.join(cleaned_sentences)
    
    def _validate_numbers(self, text: str, chunks: List[Dict]) -> str:
        import re
        
        number_claims = re.findall(r'(\w+)\s+(?:has|have|is|are)\s+(\d+%?)\s+since\s+(\w+)', text, re.IGNORECASE)
        
        if len(number_claims) > 3:
            logger.warning(f"[POST_PROCESS] Found {len(number_claims)} similar number patterns - possible repetition loop")
            
            unique_claims = {}
            for claim in number_claims:
                key = (claim[0].lower(), claim[2].lower())
                if key not in unique_claims:
                    unique_claims[key] = claim[1]
            
            if len(unique_claims) == 1 and len(number_claims) > 2:
                logger.warning("[POST_PROCESS] Repetition loop detected in number claims - removing duplicates")
                
                first_occurrence = None
                for i, match in enumerate(re.finditer(
                    r'(\w+)\s+(?:has|have|is|are)\s+(\d+%?)\s+since\s+(\w+)', text, re.IGNORECASE
                )):
                    if i == 0:
                        first_occurrence = match
                    elif i > 0:
                        text = text[:match.start()] + text[match.end():]
                        break
        
        return text
    
    def _sentence_similarity(self, s1: str, s2: str) -> float:
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        if len(chunks) <= 1:
            return chunks
        
        seen_content = set()
        unique_chunks = []
        
        for chunk in chunks:
            content = chunk.get("content", "")
            content_preview = content[:200].lower().strip()
            
            if content_preview not in seen_content:
                seen_content.add(content_preview)
                unique_chunks.append(chunk)
            else:
                logger.debug(f"[DEDUP] Removed duplicate chunk: {chunk.get('id', 'unknown')}")
        
        if len(unique_chunks) < len(chunks):
            logger.info(f"[DEDUP] Removed {len(chunks) - len(unique_chunks)} duplicate chunks")
        
        return unique_chunks
    
    def formatter_node(self, state: RAGState) -> RAGState:
        """
        Node 4: Formatter
        Extract and format citations from synthesis
        """
        logger.info(f"Formatter: Processing citations...")
        
        try:
            synthesis = state["synthesis"]
            chunks = state.get("retrieved_chunks", [])
            
            import re
            citations = []
            
            matches = re.findall(r'\[SECTION (\d+)\]', synthesis)
            
            for section_num in set(matches):
                try:
                    idx = int(section_num) - 1
                    if 0 <= idx < len(chunks):
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
            
            if not citations and chunks:
                logger.warning("[CITATIONS] No citations found in answer, adding top chunks as citations")
                for i, chunk in enumerate(chunks[:3]):
                    metadata = chunk.get("metadata", {})
                    citations.append({
                        "section": str(i + 1),
                        "page": metadata.get("page_number", "N/A"),
                        "topic": metadata.get("section", "N/A"),
                        "similarity_score": round(chunk.get("similarity", 0), 3),
                        "source_document": metadata.get("source", "Unknown")
                    })
            
            state["citations"] = citations
            logger.info(f"[CITATIONS] Extracted {len(citations)} citations")
            
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
        
        request_id = str(uuid.uuid4())
        initial_state: RAGState = {
            "query": query,
            "document_id": document_id,
            "request_id": request_id,
            "router_decision": None,
            "retrieved_chunks": [],
            "synthesis": "",
            "citations": [],
            "execution_time": 0.0,
            "error": None,
            "is_retrospective": False
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
        
        request_logger.log_request_complete(
            request_id,
            execution_time * 1000,
            success=final_state.get("error") is None,
            error=final_state.get("error")
        )
        
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