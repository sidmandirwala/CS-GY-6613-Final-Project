import os
from typing import List, Dict, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import json
import hashlib

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np
from loguru import logger

load_dotenv()

logger.add("feature_pipeline.log", rotation="500 MB")

class ContentType(str, Enum):
    ARTICLE = "article"
    CODE = "code"
    POST = "post"
    PROFILE = "profile"
    UNKNOWN = "unknown"

@dataclass
class ProcessedChunk:
    content: str
    content_type: ContentType
    metadata: Dict
    embedding: Optional[List[float]] = None
    
class ContentClassifier:
    def classify_content(self, text: str) -> ContentType:
        code_indicators = [
            "def ", "class ", "import ", "from ", "return",
            "{", "}", "//", "/", "public ", "private ",
            "function", "var ", "let ", "const ", "#include",
            "package ", "using ", "@", "->", "=>",
        ]
        
        article_indicators = [
            "Abstract:", "Introduction:", "Conclusion:",
            "In this article", "we discuss", "research shows",
            "according to", "published", "study", "author",
            "argues", "examines", "investigates"
        ]
        
        profile_indicators = [
            "experience:", "skills:", "education:", 
            "linkedin", "profile", "summary", 
            "professional", "job title", "work history"
        ]
        
        code_score = sum(1 for indicator in code_indicators if indicator in text.lower())
        article_score = sum(1 for indicator in article_indicators if indicator in text.lower())
        profile_score = sum(1 for indicator in profile_indicators if indicator in text.lower())
        
        lines = text.split('\n')
        indentation_pattern = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
        code_score += indentation_pattern * 0.5
        
        if code_score > article_score * 2 and code_score > profile_score:
            return ContentType.CODE
        elif profile_score > code_score and profile_score > article_score:
            return ContentType.PROFILE
        elif article_score > code_score and article_score > profile_score:
            return ContentType.ARTICLE
        elif len(text.split()) > 20: 
            return ContentType.POST
        else:
            return ContentType.UNKNOWN

class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def generate(self, text: str) -> List[float]:
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []

class MongoDBHandler:
    def __init__(self, uri: str, db_name: str, collection_name: str):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.processed_collection = self.db[f"{collection_name}_processed"]
        
    def get_unprocessed_documents(self, max_documents: int = 100) -> List[Dict]:
        """Retrieve unprocessed documents with a limit."""
        query = {
            "processed": {"$ne": True}
        }
        return list(self.collection.find(query).limit(max_documents))
    
    def update_document_status(self, doc_id: str, processed_data_file: str):
        """Update document status and store processed data reference."""
        try:
            update_data = {
                "processed": True,
                "processed_at": datetime.utcnow(),
                "processed_file": processed_data_file
            }
            
            self.collection.update_one(
                {"_id": doc_id},
                {"$set": update_data}
            )
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")

class ChunkProcessor:
    def __init__(self):
        self.classifier = ContentClassifier()
        self.embedding_generator = EmbeddingGenerator()
        
    def process_chunk(self, content: str, metadata: Dict) -> ProcessedChunk:
        # Clean content
        cleaned_content = self._clean_text(content)
        
        # Classify content
        content_type = self.classifier.classify_content(cleaned_content)
        
        # Generate embedding
        embedding = self.embedding_generator.generate(cleaned_content)
        
        return ProcessedChunk(
            content=cleaned_content,
            content_type=content_type,
            metadata=metadata,
            embedding=embedding
        )
    
    @staticmethod
    def _clean_text(text: str) -> str:
        text = str(text).strip()
        text = ' '.join(text.split()) 
        return text

class FeaturePipeline:
    def __init__(self, mongo_uri: str, source_configurations: List[Dict], output_dir: str):
        """
        Initialize pipeline with multiple source configurations
        
        :param mongo_uri: MongoDB connection URI
        :param source_configurations: List of dictionaries with source details
        :param output_dir: Directory to save processed files
        """
        self.source_configurations = source_configurations
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.client = MongoClient(mongo_uri)
        
    def _generate_file_name(self, doc_id: str, source: str) -> str:
        """Generate a unique filename based on document ID and source."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{source}_{doc_id}_{timestamp}.json"
    
    def _split_large_document(self, document: Dict, max_chunk_size: int = 10000) -> Dict:
        """Split large documents into smaller chunks."""
        original_chunks = document.get("chunks", [document]) if document.get("chunks") else [document]
        new_chunks = []
        
        for chunk in original_chunks:
            content = chunk.get("content", "")
            
            if len(content) > max_chunk_size:
                for i in range(0, len(content), max_chunk_size):
                    new_chunks.append({
                        "content": content[i:i+max_chunk_size],
                        "original_chunk_index": original_chunks.index(chunk)
                    })
            else:
                new_chunks.append(chunk)
        
        document["chunks"] = new_chunks
        return document
    
    def process_all_sources(self, batch_size: int = 10, max_chunk_size: int = 10000, max_documents_per_source: int = 100):
        """Process documents from all configured sources."""
        total_processed_count = 0
        total_error_count = 0
        
        logger.info(f"Starting processing for {len(self.source_configurations)} sources")
        
        for source_config in self.source_configurations:
            db_name = source_config['db_name']
            collection_name = source_config['collection_name']
            source_name = source_config.get('source_name', collection_name)
            
            logger.info(f"Processing source: {source_name}")
            
            mongodb = MongoDBHandler(
                uri=source_config['mongo_uri'], 
                db_name=db_name, 
                collection_name=collection_name
            )
            
            processor = ChunkProcessor()
            
            documents = mongodb.get_unprocessed_documents(max_documents_per_source)
            logger.info(f"Found {len(documents)} unprocessed documents in {source_name}")
            
            processed_count = 0
            error_count = 0
            
            for i, doc in enumerate(documents):
                if i % batch_size == 0:
                    logger.info(f"Processing batch {i // batch_size + 1} for {source_name}")
                
                try:
                    doc_size = len(json.dumps(doc))
                    logger.info(f"Document {doc['_id']} from {source_name} size: {doc_size} bytes")
                    
                    doc = self._split_large_document(doc, max_chunk_size)
                    
                    processed_data = self._process_single_document(
                        document=doc, 
                        processor=processor, 
                        source_name=source_name
                    )
                    
                    output_filename = self._generate_file_name(doc['_id'], source_name)
                    output_filepath = os.path.join(self.output_dir, output_filename)
                    
                    self._save_processed_data(output_filepath, processed_data)
                    
                    mongodb.update_document_status(doc['_id'], output_filename)
                    
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing document {doc['_id']} from {source_name}: {e}")
                    error_count += 1
            
            logger.success(f"Completed {source_name}. Processed: {processed_count}, Errors: {error_count}")
            
            total_processed_count += processed_count
            total_error_count += error_count
        
        logger.info(f"Overall Processing Complete. Total Processed: {total_processed_count}, Total Errors: {total_error_count}")
    
    def _process_single_document(self, document: Dict, processor: ChunkProcessor, source_name: str) -> Dict:
        """Process a single document and its chunks."""
        chunks = document.get("chunks", [])
        processed_chunks = []
        
        for chunk in chunks:
            metadata = {
                "doc_id": document["_id"],
                "source": source_name,
                "processed_at": datetime.utcnow().isoformat()
            }
            
            try:
                processed_chunk = processor.process_chunk(
                    content=chunk.get("content", ""),
                    metadata=metadata
                )
                
                processed_chunks.append({
                    "content": processed_chunk.content,
                    "content_type": processed_chunk.content_type,
                    "metadata": processed_chunk.metadata,
                    "embedding": processed_chunk.embedding
                })
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
        
        return {
            "chunks": processed_chunks,
            "total_chunks": len(processed_chunks),
            "content_type_distribution": self._get_content_distribution(processed_chunks)
        }
    
    def _get_content_distribution(self, chunks: List[Dict]) -> Dict:
        """Calculate distribution of content types."""
        distribution = {}
        for chunk in chunks:
            content_type = chunk["content_type"]
            distribution[content_type] = distribution.get(content_type, 0) + 1
        return distribution
    
    def _save_processed_data(self, filepath: str, processed_data: Dict):
        """Save processed data to file system."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving processed data to {filepath}: {e}")

def main():
    MONGO_URI = os.getenv("BASE_MONGO_URI", "mongodb://localhost:27017/")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "processed_data")
    
    SOURCE_CONFIGURATIONS = [
        {
            "mongo_uri": MONGO_URI,
            "db_name": "github_scraper",
            "collection_name": "repositories",
            "source_name": "GitHub"
        },
        {
            "mongo_uri": MONGO_URI,
            "db_name": "medium_scraper",
            "collection_name": "repositories",
            "source_name": "Medium"
        },
        {
            "mongo_uri": MONGO_URI,
            "db_name": "linkedin_scraper",
            "collection_name": "profiles",
            "source_name": "LinkedIn"
        }
    ]
    
    logger.info("Multi-Source Feature Pipeline Starting...")
    
    try:
        # Initialize pipeline with multiple sources
        pipeline = FeaturePipeline(
            mongo_uri=MONGO_URI,
            source_configurations=SOURCE_CONFIGURATIONS,
            output_dir=OUTPUT_DIR
        )
        
        # Process documents from all sources
        pipeline.process_all_sources(
            batch_size=10,           
            max_chunk_size=10000,   
            max_documents_per_source=100  
        )
        
        logger.success("Multi-Source Feature Pipeline Completed Successfully")
    
    except Exception as e:
        logger.error(f"Fatal error in Feature Pipeline: {e}")

if __name__ == "__main__":
    main()
