import json
import numpy as np
from sentence_transformers import SentenceTransformer
from app.db.vectorstore import create_vectorstore  # Import the vectorstore creator

class RAGChain:
    def __init__(self, template_file: str):
        self.template_file = template_file
        self.templates = self.load_templates()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.embeddings = None
        self.build_index()
    
    def load_templates(self):
        with open(self.template_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def build_index(self):
        # Extract the "emotion" fields from the templates and compute embeddings.
        sentences = [
            f"Emotion: {t['emotion']}. Examples: {' '.join(t['templates'][:3])}" 
            for t in self.templates
        ]
        self.embeddings = self.embedder.encode(sentences)
        # Create the FAISS index using the separate vectorstore module.
        self.index = create_vectorstore(self.embeddings)
    
    def retrieve_templates(self, query: str, top_k: int = 2):
        # Encode the query using the same embedder.
        query_vec = self.embedder.encode([query])
        # Search the FAISS index.
        distances, indices = self.index.search(np.array(query_vec, dtype='float32'), top_k)
        # Return the corresponding templates.
        return [self.templates[i] for i in indices[0]]
