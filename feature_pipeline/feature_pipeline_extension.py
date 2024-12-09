import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams, PointStruct
from loguru import logger

# Load environment variables
load_dotenv()

@dataclass
class SearchResult:
    content: str
    content_type: str
    metadata: Dict
    score: float
    
class OpenAIEmbedding:
    def __init__(self, api_key: Optional[str] = None):
        try:
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.model = "text-embedding-3-small"
            self.dimension = 384  # Dimension of text-embedding-3-small embeddings
        except AuthenticationError:
            print("Your API-KEY is not correct!")
            raise SystemExit(1)
        
    def create_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text using OpenAI API."""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except AuthenticationError:
            print("Your API-KEY is not correct!")
            raise SystemExit(1)
        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {e}")
            raise

class QdrantStorage:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection_name: str = "content_vectors"
    ):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.embedding_dimension = 384  # OpenAI embedding dimension
        
    def initialize_collection(self, recreate: bool = False):
        """Initialize or recreate the vector collection."""
        try:
            if recreate:
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection {self.collection_name} recreated")
            else:
                collections = self.client.get_collections()
                if not any(c.name == self.collection_name for c in collections.collections):
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=self.embedding_dimension,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"Collection {self.collection_name} created")
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise
            
    def store_vectors(self, vectors: List[Dict]):
        """Store vectors in Qdrant."""
        try:
            points = []
            for idx, vector_data in enumerate(vectors):
                point = PointStruct(
                    id=idx,
                    vector=vector_data["embedding"],
                    payload={
                        "content": vector_data["content"],
                        "content_type": vector_data["content_type"],
                        "metadata": vector_data["metadata"]
                    }
                )
                points.append(point)
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Stored {len(points)} vectors in Qdrant")
        except Exception as e:
            logger.error(f"Error storing vectors: {e}")
            raise

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            
            return [
                SearchResult(
                    content=r.payload["content"],
                    content_type=r.payload["content_type"],
                    metadata=r.payload["metadata"],
                    score=r.score
                )
                for r in results
            ]
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            raise

class RAGSystem:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "content_vectors"
    ):
        self.embedding_engine = OpenAIEmbedding(api_key=openai_api_key)
        self.vector_store = QdrantStorage(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name
        )
        
    def initialize(self, recreate_collection: bool = False):
        """Initialize the RAG system."""
        self.vector_store.initialize_collection(recreate=recreate_collection)
        
    def load_processed_data(self, data_dir: str):
        """Load and store processed data from feature pipeline."""
        vectors = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(data_dir, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    vectors.extend(data['chunks'])
        
        self.vector_store.store_vectors(vectors)
        
    def query(
        self,
        question: str,
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[SearchResult]:
        """Query the RAG system."""
        # Generate embedding for question
        query_embedding = self.embedding_engine.create_embedding(question)
        
        # Search for similar content
        results = self.vector_store.search(
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold
        )
        
        return results

def main():
    # Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    QDRANT_URL = "http://localhost:6333"
    PROCESSED_DATA_DIR = "processed_data"
    
    # Initialize RAG system
    rag_system = RAGSystem(
        openai_api_key=OPENAI_API_KEY,
        qdrant_url=QDRANT_URL
    )
    
    # Initialize collections
    rag_system.initialize(recreate_collection=True)
    
    # Load processed data
    rag_system.load_processed_data(PROCESSED_DATA_DIR)
    
    # Example query
    question = "How to implement a binary search tree?"
    results = rag_system.query(question, limit=3)
    
    # Print results
    print(f"\nResults for question: {question}\n")
    for i, result in enumerate(results, 1):
        print(f"Result {i} ({result.content_type}) - Score: {result.score:.3f}")
        print(f"Content: {result.content[:200]}...")
        print(f"Metadata: {result.metadata}")
        print()

if __name__ == "__main__":
    main()