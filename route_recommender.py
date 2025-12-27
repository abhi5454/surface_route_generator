"""
Route recommendation engine combining semantic search with filters.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from data_processor import RouteData
from embedding_generator import EmbeddingGenerator
from vector_store import VectorStore
from osm_enricher import OSMEnricher
from query_processor import QueryProcessor
from geocoder import LocationGeocoder


@dataclass
class RouteRecommendation:
    """Container for route recommendation with score"""
    route_id: str
    route_name: str
    distance_km: float
    elevation_gain: float
    surface_type: str
    amenities: List[Dict]
    similarity_score: float
    metadata: Dict
    explanation: str
    start_location: str = "Unknown"
    end_location: str = "Unknown"
    start_coords: tuple = None
    end_coords: tuple = None


class RouteRecommender:
    """Recommend routes based on user queries"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
        osm_enricher: Optional[OSMEnricher] = None,
        query_processor: Optional[QueryProcessor] = None
    ):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.osm_enricher = osm_enricher or OSMEnricher()
        self.query_processor = query_processor
        self.geocoder = LocationGeocoder()
        
        if query_processor:
            print("Route Recommender initialized with Claude query processing")
        else:
            print("Route Recommender initialized (query processing disabled)")
    
    
    def search_routes(
        self,
        query: str,
        n_results: int = 5,
        use_llm: bool = True
    ) -> List[RouteRecommendation]:
        """Search for routes using natural language query"""
        
        # Parse query parameters
        location_coords = None
        location_name = None
        
        if use_llm and self.query_processor:
            try:
                query_params = self.query_processor.parse_query(query)
                print(f"Parsed query: {query_params.get('distance_km', 'any')} km, "
                      f"Type: {query_params.get('activity_type', 'any')}")
                
                # Check for location
                location_name = query_params.get('location')
                if location_name:
                    print(f"Filtering by location: {location_name}")
                    location_coords = self.geocoder.forward_geocode(location_name)
                    if location_coords:
                        print(f"  Found coordinates: {location_coords}")
                    else:
                        print(f"  Could not geocode location: {location_name}")
                        
            except Exception as e:
                print(f"Query parsing failed, using fallback: {e}")
                query_params = self._simple_parse(query)
        else:
            query_params = self._simple_parse(query)
        
        # Generate embedding for query
        query_embedding = self.embedding_generator.generate_single_embedding(query)
        
        # Determine how many results to fetch initially
        # If filtering by location, fetch more results to allow for filtering
        fetch_n = n_results * 5 if location_coords else n_results
        
        # Search
        results = self.vector_store.search_by_filters(
            query_embedding,
            distance_range=query_params.get('distance_range'),
            activity_types=None,
            n_results=fetch_n
        )
        
        # Convert to recommendations
        recommendations = self._process_results(results, query_params)
        
        # Filter by location if applicable
        if location_coords:
            filtered_recs = []
            max_distance_km = 30.0  # Max distance from search location
            
            for rec in recommendations:
                if rec.start_coords:
                    dist = self.geocoder.calculate_distance(location_coords, rec.start_coords)
                    if dist <= max_distance_km:
                        # Update explanation to mention proximity
                        rec.explanation += f" ({dist:.1f} km from {location_name})"
                        filtered_recs.append(rec)
            
            if filtered_recs:
                recommendations = filtered_recs
                print(f"Filtered to {len(recommendations)} routes near {location_name}")
            else:
                print(f"No routes found near {location_name}, showing original results")
        
        # Return top N
        return recommendations[:n_results]
    
    def _process_results(
        self,
        results: Dict,
        query_params: Dict
    ) -> List[RouteRecommendation]:
        """Process search results into recommendations"""
        recommendations = []
        
        # Safety checks for empty or malformed results
        if not results.get('ids') or not results['ids'][0]:
            return recommendations
        
        ids = results['ids'][0]
        metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
        documents = results.get('documents', [[]])[0] if results.get('documents') else []
        distances = results.get('distances', [[]])[0] if results.get('distances') else []
        
        for i, route_id in enumerate(ids):
            # Safely get metadata, document, and distance
            metadata = metadatas[i] if i < len(metadatas) else {}
            document = documents[i] if i < len(documents) else ""
            distance = distances[i] if i < len(distances) else 0.5
            similarity = 1 - distance  # Convert distance to similarity
            
            # Extract metadata
            distance_km = float(metadata.get('distance_km', 0))
            elevation_gain = float(metadata.get('elevation_gain', 0))
            route_name = metadata.get('name', 'Untitled')
            
            # Generate explanation
            explanation_parts = [
                f"{distance_km:.1f} km route",
            ]
            
            if elevation_gain > 100:
                explanation_parts.append(f"with {elevation_gain:.0f}m elevation gain")
            
            if elevation_gain < 50:
                explanation_parts.append("(mostly flat)")
            
            explanation = " ".join(explanation_parts)
            
            # Extract start/end coordinates from metadata
            start_coords = None
            end_coords = None
            try:
                start_lat = metadata.get('start_lat')
                start_lon = metadata.get('start_lon')
                end_lat = metadata.get('end_lat')
                end_lon = metadata.get('end_lon')
                
                if start_lat and start_lon and start_lat != 'None' and start_lon != 'None':
                    start_coords = (float(start_lat), float(start_lon))
                if end_lat and end_lon and end_lat != 'None' and end_lon != 'None':
                    end_coords = (float(end_lat), float(end_lon))
            except (ValueError, TypeError):
                pass
            
            recommendation = RouteRecommendation(
                route_id=route_id,
                route_name=route_name,
                distance_km=distance_km,
                elevation_gain=elevation_gain,
                surface_type="unknown",  # Will be enriched if needed
                amenities=[],
                similarity_score=similarity,
                metadata=metadata,
                explanation=explanation,
                start_coords=start_coords,
                end_coords=end_coords
            )
            
            recommendations.append(recommendation)
        
        # Sort by similarity
        recommendations.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return recommendations
    
    def _simple_parse(self, query: str) -> Dict:
        """Simple query parsing without LLM"""
        import re
        
        params = {
            'distance_km': None,
            'distance_range': None,
            'activity_type': None,
            'amenities_required': [],
            'surface_preferences': []
        }
        
        # Extract distance
        dist_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:km|kilometer)', query.lower())
        if dist_match:
            dist = float(dist_match.group(1))
            params['distance_km'] = dist
            params['distance_range'] = (dist * 0.8, dist * 1.2)
        
        return params
    
    def format_recommendations(
        self,
        recommendations: List[RouteRecommendation],
        show_details: bool = True
    ) -> str:
        """Format recommendations as human-readable text"""
        if not recommendations:
            return "No routes found matching your criteria."
        
        output_lines = [
            f"\n{'='*70}",
            f"Found {len(recommendations)} route recommendations",
            f"{'='*70}\n"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            output_lines.append(f"{i}. {rec.route_name}")
            output_lines.append(f"   {rec.explanation}")
            
            # Show location information
            if rec.start_location and rec.start_location != "Unknown":
                output_lines.append(f"    Start: {rec.start_location}")
            if rec.end_location and rec.end_location != "Unknown":
                output_lines.append(f"    End:   {rec.end_location}")
            # if rec.elevation_gain and rec.elevation_gain != "Unknown":
            output_lines.append(f"    Elevation Gain: {int(rec.elevation_gain)}m")
            
            output_lines.append(f"    Similarity: {rec.similarity_score:.1%} (Semantic match to your query)")
            
            if show_details:
                output_lines.append(f"    Route ID: {rec.route_id}")
            
            output_lines.append("")
        
        return "\n".join(output_lines)


if __name__ == "__main__":
    # Test route recommender
    from data_processor import ActivityParser
    import os
    
    print("Testing Route Recommender\n")
    
    # Parse activities
    parser = ActivityParser()
    routes = parser.parse_all_activities(limit=10)
    
    # Generate embeddings
    generator = EmbeddingGenerator()
    embeddings = generator.generate_route_embeddings(routes)
    
    # Create vector store
    store = VectorStore()
    store.clear()  # Clear for testing
    store.add_routes(routes, embeddings)
    
    # Initialize recommender
    use_claude = bool(os.getenv("ANTHROPIC_API_KEY"))
    if use_claude:
        query_proc = QueryProcessor()
    else:
        query_proc = None
        print("Note: Claude query processing disabled (no API key)")
    
    recommender = RouteRecommender(
        vector_store=store,
        embedding_generator=generator,
        query_processor=query_proc
    )
    
    # Test searches
    test_queries = [
        "5 km run",
        "10 kilometer loop",
        "short walk"
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
        
        recommendations = recommender.search_routes(
            query,
            n_results=3,
            use_llm=use_claude
        )
        
        print(recommender.format_recommendations(recommendations))
