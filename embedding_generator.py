"""
Embedding generator using sentence-transformers with PyTorch.
"""
import torch
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from tqdm import tqdm

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_LENGTH,
    DEVICE,
    TORCH_DTYPE
)
from data_processor import RouteData


class EmbeddingGenerator:
    """Generate embeddings for route descriptions using sentence-transformers"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        print(f"Loading embedding model: {model_name} on {DEVICE}")
        
        # Load model with PyTorch device configuration
        self.model = SentenceTransformer(model_name, device=DEVICE)
        
        # Configure for GPU if available
        if DEVICE == "cuda":
            self.model = self.model.half()  # Use FP16 for faster inference
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        with torch.no_grad():
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
        return embedding
    
    def generate_batch_embeddings(
        self,
        texts: List[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for a batch of texts"""
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                normalize_embeddings=True
            )
        return embeddings
    
    def generate_route_embeddings(
        self,
        routes: List[RouteData],
        batch_size: int = EMBEDDING_BATCH_SIZE
    ) -> np.ndarray:
        """Generate embeddings for a list of routes"""
        print(f"Generating embeddings for {len(routes)} routes...")
        
        # Extract descriptions
        descriptions = [route.get_description() for route in routes]
        
        # Generate embeddings in batches
        embeddings = self.generate_batch_embeddings(
            descriptions,
            batch_size=batch_size,
            show_progress=True
        )
        
        print(f"Generated {len(embeddings)} embeddings of dimension {self.embedding_dim}")
        return embeddings
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        route_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between query and route embeddings"""
        # Embeddings are already normalized, so dot product = cosine similarity
        similarities = np.dot(route_embeddings, query_embedding)
        return similarities


if __name__ == "__main__":
    # Test embedding generation
    from data_processor import ActivityParser
    
    print("Testing Embedding Generator\n")
    print(f"Using device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    
    if DEVICE == "cuda":
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Parse some activities
    parser = ActivityParser()
    routes = parser.parse_all_activities(limit=5)
    
    # Generate embeddings
    generator = EmbeddingGenerator()
    embeddings = generator.generate_route_embeddings(routes)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Sample embedding (first 10 dims): {embeddings[0][:10]}")
    
    # Test similarity
    query = "5km run with moderate elevation"
    query_embedding = generator.generate_single_embedding(query)
    
    similarities = generator.compute_similarity(query_embedding, embeddings)
    
    print(f"\nQuery: {query}")
    print(f"Similarities: {similarities}")
    
    # Show top match
    top_idx = np.argmax(similarities)
    print(f"\nTop match:")
    print(f"  {routes[top_idx].get_description()}")
    print(f"  Similarity: {similarities[top_idx]:.3f}")
