"""
RAG retriever for maintenance knowledge.
Handles querying the vector index and returning relevant context.
"""
from typing import List, Dict
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient

from src.config import Config


class MaintenanceRetriever:
    """Retrieves relevant maintenance documentation."""
    
    def __init__(
        self,
        collection_name: str = None,
        top_k: int = None
    ):
        """
        Initialize retriever.
        
        Args:
            collection_name: Qdrant collection name
            top_k: Number of results to retrieve
        """
        self.collection_name = collection_name or Config.QDRANT_COLLECTION
        self.top_k = top_k or Config.RAG_TOP_K
        
        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name=Config.EMBEDDING_MODEL
        )
        Settings.embed_model = self.embed_model
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            host=Config.QDRANT_HOST,
            port=Config.QDRANT_PORT
        )
        
        # Create vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name
        )
        
        # Create index
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            
        Returns:
            List of retrieved documents with metadata
        """
        # Query the index
        retriever = self.index.as_retriever(similarity_top_k=self.top_k)
        nodes = retriever.retrieve(query)
        
        # Format results
        results = []
        for node in nodes:
            results.append({
                'text': node.node.text,
                'score': node.score,
                'metadata': node.node.metadata,
                'node_id': node.node.node_id
            })
        
        return results
    
    def query(self, query: str) -> str:
        """
        Query the index and get relevant context.
        
        Args:
            query: Question or search query
            
        Returns:
            Formatted context from retrieved documents
        """
        docs = self.retrieve(query)
        return self.format_context(docs)
    
    def retrieve_for_symptoms(self, symptoms: Dict) -> List[Dict]:
        """
        Retrieve maintenance procedures for specific symptoms.
        
        Args:
            symptoms: Dictionary of detected symptoms
            
        Returns:
            Relevant maintenance procedures
        """
        # Build query from symptoms
        query_parts = []
        
        if symptoms.get('high_vibration'):
            query_parts.append("high vibration troubleshooting")
        
        if symptoms.get('high_temperature'):
            query_parts.append("high temperature corrective actions")
        
        if symptoms.get('low_pressure_ratio'):
            query_parts.append("low pressure ratio maintenance")
        
        if symptoms.get('high_rul_risk'):
            query_parts.append("preventive maintenance schedule")
        
        # Combine queries
        query = " ".join(query_parts) if query_parts else "general maintenance procedures"
        
        return self.retrieve(query)
    
    def get_safety_procedures(self) -> List[Dict]:
        """
        Retrieve safety procedures.
        
        Returns:
            Safety-related documentation
        """
        return self.retrieve("safety procedures warnings precautions")
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant documentation found."
        
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(
                f"[Source {i}] (Relevance: {doc['score']:.2f})\n"
                f"{doc['text']}\n"
            )
        
        return "\n".join(context_parts)
