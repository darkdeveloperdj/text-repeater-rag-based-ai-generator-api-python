from app.chains.love_rag_chain import RAGChain as LoveRAGChain
from app.chains.quotes_rag_chain import QuotesRAGChain
from typing import Optional

class BaseAgent:
    def __init__(self, love_templates_file: str = None, quotes_templates_file: str = None):
        if love_templates_file:
            self.love_rag_chain = LoveRAGChain(love_templates_file)
        else:
            self.love_rag_chain = None  # Set to None if no file is provided
        
        if quotes_templates_file:
            self.quotes_rag_chain = QuotesRAGChain(quotes_templates_file)
        else:
            self.quotes_rag_chain = None  # Set to None if no file is provided

    def retrieve_templates(self, query: str, top_k: int = 2, req_type: str = "love"):
        if req_type == "love" and self.love_rag_chain:
            return self.love_rag_chain.retrieve_templates(query, top_k)
        elif req_type == "quote" and self.quotes_rag_chain:
            return self.quotes_rag_chain.retrieve_quotes(query, top_k)
        else:
            raise ValueError(f"Unsupported type or chain not initialized: {req_type}")

