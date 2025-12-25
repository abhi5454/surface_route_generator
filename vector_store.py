"""
Vector store using ChromaDB for route storage and similarity search.
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
import numpy as np
from pathlib import Path

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    DEFAULT_NUM_RESULTS,
    MIN_SIMILARITY_SCORE
)
from data_processor import RouteData


class VectorStore:
    """ChromaDB-based vector store for routes"""
    
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        """Initialize ChromaDB client"""
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"description": "Strava route embeddings"}
        )
        
        print(f"Vector store initialized with {self.collection.count()} routes")
    
    def add_routes(
        self,
        routes: List[RouteData],
        embeddings: np.ndarray,
        metadata_extra: Optional[List[Dict]] = None
    ):
        """Add routes to the vector store"""
        if len(routes) != len(embeddings):
            raise ValueError("Number of routes must match number of embeddings")
        
        # Prepare data for ChromaDB
        ids = [route.activity_id for route in routes]
        embeddings_list = embeddings.tolist()
        documents = [route.get_description() for route in routes]
        
        # Prepare metadata
        metadatas = []
        for i, route in enumerate(routes):
            metadata = route.to_dict()
            
            # Add extra metadata if provided
            if metadata_extra and i < len(metadata_extra):
                metadata.update(metadata_extra[i])
            
            # Convert all values to strings for ChromaDB
            metadata = {k: str(v) if v is not None else "" for k, v in metadata.items()}
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"Added {len(routes)} routes to vector store")
        print(f"Total routes in store: {self.collection.count()}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        n_results: int = DEFAULT_NUM_RESULTS,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """Search for similar routes"""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
            where_document=where_document
        )
        
        return results
    
    def search_by_filters(
        self,
        query_embedding: np.ndarray,
        distance_range: Optional[tuple] = None,  # (min_km, max_km)
        activity_types: Optional[List[str]] = None,
        n_results: int = DEFAULT_NUM_RESULTS
    ) -> Dict:
        """Search with specific filters"""
        where_clause = {}
        
        # Note: ChromaDB metadata filters work on strings
        # We'll need to convert numeric comparisons
        
        if activity_types:
            where_clause["activity_type"] = {"$in": activity_types}
        
        # For distance filtering, we'll need to filter results post-query
        # since ChromaDB doesn't support numeric range queries well on string metadata
        results = self.search(
            query_embedding,
            n_results=n_results * 2,  # Get more results to filter
            where=where_clause if where_clause else None
        )
        
        # Post-filter by distance if specified
        if distance_range and results['ids']:
            filtered_results = self._filter_by_distance(results, distance_range)
            return filtered_results
        
        return results
    
    def _filter_by_distance(
        self,
        results: Dict,
        distance_range: tuple
    ) -> Dict:
        """Filter results by distance range"""
        min_km, max_km = distance_range
        
        filtered_ids = []
        filtered_embeddings = []
        filtered_documents = []
        filtered_metadatas = []
        filtered_distances = []
        
        # Safety check for empty results
        if not results.get('metadatas') or not results['metadatas'][0]:
            return results
            
        metadatas = results['metadatas'][0]
        ids = results['ids'][0]
        documents = results.get('documents', [[]])[0]
        distances = results.get('distances', [[]])[0]
        embeddings = results.get('embeddings', [[]])[0] if results.get('embeddings') else []
        
        for i, metadata in enumerate(metadatas):
            try:
                distance = float(metadata.get('distance_km', 0))
                if min_km <= distance <= max_km:
                    filtered_ids.append(ids[i])
                    filtered_metadatas.append(metadata)
                    if i < len(documents):
                        filtered_documents.append(documents[i])
                    if i < len(distances):
                        filtered_distances.append(distances[i])
                    if embeddings and i < len(embeddings):
                        filtered_embeddings.append(embeddings[i])
            except (ValueError, TypeError) as e:
                print(f"Error filtering route: {e}")
                continue
        
        return {
            'ids': [filtered_ids],
            'embeddings': [filtered_embeddings] if filtered_embeddings else None,
            'documents': [filtered_documents],
            'metadatas': [filtered_metadatas],
            'distances': [filtered_distances]
        }
    
    def get_route_by_id(self, activity_id: str) -> Optional[Dict]:
        """Retrieve a specific route by ID"""
        result = self.collection.get(
            ids=[activity_id],
            include=['embeddings', 'documents', 'metadatas']
        )
        
        if result['ids']:
            return {
                'id': result['ids'][0],
                'embedding': result['embeddings'][0] if 'embeddings' in result else None,
                'document': result['documents'][0],
                'metadata': result['metadatas'][0]
            }
        return None
    
    def count(self) -> int:
        """Get total number of routes in store"""
        return self.collection.count()
    
    def clear(self):
        """Clear all routes from the store"""
        self.client.delete_collection(CHROMA_COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"description": "Strava route embeddings"}
        )
        print("Vector store cleared")
    
    def export_metadata(self) -> List[Dict]:
        """Export all route metadata"""
        results = self.collection.get(
            include=['documents', 'metadatas']
        )
        
        return [
            {
                'id': results['ids'][i],
                'document': results['documents'][i],
                'metadata': results['metadatas'][i]
            }
            for i in range(len(results['ids']))
        ]


if __name__ == "__main__":
    # Test vector store
    from data_processor import ActivityParser
    from embedding_generator import EmbeddingGenerator
    
    print("Testing Vector Store\n")
    
    # Parse activities
    parser = ActivityParser()
    routes = parser.parse_all_activities(limit=5)
    
    # Generate embeddings
    generator = EmbeddingGenerator()
    embeddings = generator.generate_route_embeddings(routes)
    
    # Initialize vector store
    store = VectorStore()
    
    # Clear existing data for testing
    if store.count() > 0:
        print(f"Clearing {store.count()} existing routes")
        store.clear()
    
    # Add routes
    store.add_routes(routes, embeddings)
    
    # Test search
    query = "5 kilometer run with hills"
    query_embedding = generator.generate_single_embedding(query)
    
    print(f"\nSearching for: {query}")
    results = store.search(query_embedding, n_results=3)
    
    print(f"\nTop 3 results:")
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\n{i+1}. {doc}")
        print(f"   Distance: {metadata['distance_km']} km")
        print(f"   Similarity score: {1 - distance:.3f}")
