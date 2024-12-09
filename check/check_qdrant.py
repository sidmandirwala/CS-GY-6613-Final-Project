from qdrant_client import QdrantClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_API_KEY = None  
COLLECTION_NAME = "qna_collection_self"

def check_qdrant_points():
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,
    )
    
    try:
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        
        point_count = collection_info.points_count
        logger.info(f"Collection '{COLLECTION_NAME}' has {point_count} points.")
    except Exception as e:
        logger.error(f"Error accessing Qdrant: {e}")

if __name__ == "__main__":
    check_qdrant_points()
