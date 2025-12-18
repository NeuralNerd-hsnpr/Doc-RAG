import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


class QueryProcessor:
    def __init__(self):
        self.expansion_patterns = {
            "topic": ["main topic", "subject", "theme", "focus", "content"],
            "summary": ["overview", "summary", "main points", "key points"],
            "what": ["describe", "explain", "tell me about"],
            "how": ["method", "process", "way", "approach"],
            "why": ["reason", "cause", "purpose", "motivation"]
        }
    
    def expand_query(self, query: str) -> str:
        query_lower = query.lower().strip()
        expanded_terms = []
        
        for key, expansions in self.expansion_patterns.items():
            if key in query_lower:
                expanded_terms.extend(expansions)
        
        if expanded_terms:
            expanded_query = f"{query} {' '.join(expanded_terms[:3])}"
            logger.debug(f"Query expanded: '{query}' -> '{expanded_query}'")
            return expanded_query
        
        return query
    
    def preprocess_query(self, query: str) -> str:
        query = query.strip()
        
        if not query:
            return query
        
        if query.endswith('?'):
            query = query[:-1].strip()
        
        query = re.sub(r'\s+', ' ', query)
        
        if len(query.split()) < 3:
            expanded = self.expand_query(query)
            if expanded != query:
                query = expanded
        
        logger.debug(f"Preprocessed query: '{query}'")
        return query
    
    def create_query_variations(self, query: str) -> List[str]:
        variations = [query]
        
        query_lower = query.lower()
        
        if "topic" in query_lower or "subject" in query_lower:
            variations.extend([
                f"what is the main topic of {query}",
                f"what does {query} discuss",
                f"main theme of {query}"
            ])
        
        if "summary" in query_lower or "overview" in query_lower:
            variations.extend([
                f"what are the main points in {query}",
                f"key information about {query}",
                f"important details in {query}"
            ])
        
        variations = [v for v in variations if len(v.split()) >= 3][:3]
        
        logger.debug(f"Query variations: {variations}")
        return variations


query_processor = QueryProcessor()

