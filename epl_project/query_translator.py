import json
from slm_manager import SLMManager

class QueryTranslator:
    """
    [Sumanth RAG: Query Translation]
    Multi-Query / RAG-Fusion strategy to improve retrieval precision.
    Generates varied versions of a query to overcome semantic search limitations.
    """
    def __init__(self):
        self.slm = SLMManager()

    def generate_multi_queries(self, original_query: str, count: int = 3) -> list:
        """
        [Rule 22.2: Speculative Reasoning]
        Uses a Draft-Verify cycle to generate high-precision queries.
        """
        # Stage 1: Fast Drafting (Conceptual SLM Phase)
        draft_prompt = f"Draft {count} tactical/statistical variations for: {original_query}"
        draft_response = self.slm.query(draft_prompt, system_prompt="You are a Fast Query Drafter.")
        
        # Stage 2: Verification & Refinement (Conceptual Pro Phase)
        verify_prompt = f"Refine and verify these draft queries for EPL context: {draft_response}"
        final_response = self.slm.query(verify_prompt, system_prompt="You are a Senior Tactical Verifier.")
        
        queries = [q.strip() for q in final_response.split("\n") if q.strip()]
        
        # Always include original
        if original_query not in queries:
            queries.insert(0, original_query)
        return queries[:count+1]

class HybridRouter:
    """
    [Sumanth RAG: Dynamic Routing]
    Routes queries between Structured DB (DuckDB) and Unstructured context.
    """
    def __init__(self):
        self.slm = SLMManager()

    def route_query_type(self, query: str) -> str:
        """Determines if the query needs 'Structured SQL' or 'Semantic Context'."""
        prompt = f"""
        Query: "{query}"
        Available Tools:
        1. SQL_DB: For numerical stats, goals, league tables, xG numbers.
        2. SEMANTIC_CONTEXT: For tactical analysis, player news, injury reports, manager styles.
        
        Task: Return only 'SQL_DB' or 'SEMANTIC_CONTEXT'.
        """
        response = self.slm.query(prompt, system_prompt="You are a Strategic RAG Router.").upper()
        if "SQL" in response: return "SQL_DB"
        return "SEMANTIC_CONTEXT"

# Singleton instances
query_translator = QueryTranslator()
hybrid_router = HybridRouter()
