"""
OSM enricher for fetching surface and amenity data.
"""
import osmnx as ox
from typing import List, Dict, Tuple, Optional
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np

from config import (
    OSM_CACHE_DIR,
    AMENITY_SEARCH_RADIUS,
    SURFACE_TYPES,
    AMENITY_TYPES,
    ENABLE_CACHE
)
from data_processor import RouteData


class OSMEnricher:
    """Enrich routes with OpenStreetMap surface and amenity data"""
    
    def __init__(self, cache_dir: Path = OSM_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure OSMnx (using new settings API)
        ox.settings.use_cache = True
        ox.settings.log_console = False
        
        print("OSM Enricher initialized")
    
    def point_to_cache_key(self, lat: float, lon: float, radius: int) -> str:
        """Generate cache key for a point query"""
        return f"{lat:.6f}_{lon:.6f}_{radius}"
    
    def get_surface_data(
        self,
        coordinates: List[Tuple[float, float]],
        cache_key: Optional[str] = None
    ) -> Dict[str, float]:
        """Get surface type distribution for route"""
        if not coordinates:
            return {}
        
        # Check cache
        if ENABLE_CACHE and cache_key:
            cache_file = self.cache_dir / f"surface_{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        try:
            # Sample coordinates to avoid too many queries
            sample_size = min(50, len(coordinates))
            step = max(1, len(coordinates) // sample_size)
            sampled_coords = coordinates[::step]
            
            # Get bounding box
            lats = [c[0] for c in sampled_coords]
            lons = [c[1] for c in sampled_coords]
            
            north, south = max(lats), min(lats)
            east, west = max(lons), min(lons)
            
            # Add small buffer
            buffer = 0.01
            north += buffer
            south -= buffer
            east += buffer
            west -= buffer
            
            # Download street network
            G = ox.graph_from_bbox(
                north, south, east, west,
                network_type='all',
                simplify=True
            )
            
            # Extract surface information
            surface_counts = {}
            total_edges = 0
            
            for u, v, data in G.edges(data=True):
                surface = data.get('surface', 'unknown')
                
                # Normalize surface type
                surface_normalized = surface.lower() if surface else 'unknown'
                
                # Match to our surface types
                matched = False
                for surf_type in SURFACE_TYPES:
                    if surf_type in surface_normalized:
                        surface_counts[surf_type] = surface_counts.get(surf_type, 0) + 1
                        matched = True
                        break
                
                if not matched:
                    surface_counts['unknown'] = surface_counts.get('unknown', 0) + 1
                
                total_edges += 1
            
            # Convert to percentages
            if total_edges > 0:
                surface_distribution = {
                    surf: count / total_edges
                    for surf, count in surface_counts.items()
                }
            else:
                surface_distribution = {}
            
            # Cache the result
            if ENABLE_CACHE and cache_key:
                cache_file = self.cache_dir / f"surface_{cache_key}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(surface_distribution, f)
            
            return surface_distribution
            
        except Exception as e:
            print(f"Error fetching surface data: {e}")
            return {}
    
    def get_nearby_amenities(
        self,
        point: Tuple[float, float],
        radius: int = AMENITY_SEARCH_RADIUS,
        amenity_types: List[str] = AMENITY_TYPES
    ) -> List[Dict]:
        """Find amenities near a point"""
        lat, lon = point
        
        # Check cache
        cache_key = self.point_to_cache_key(lat, lon, radius)
        if ENABLE_CACHE:
            cache_file = self.cache_dir / f"amenities_{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        try:
            # Query OSM for amenities
            amenities = []
            
            tags = {'amenity': amenity_types}
            
            # Get POIs within radius
            pois = ox.features_from_point(
                (lat, lon),
                tags=tags,
                dist=radius
            )
            
            if not pois.empty:
                for idx, poi in pois.iterrows():
                    amenity_info = {
                        'type': poi.get('amenity', 'unknown'),
                        'name': poi.get('name', 'Unnamed'),
                        'lat': poi.geometry.centroid.y if hasattr(poi.geometry, 'centroid') else lat,
                        'lon': poi.geometry.centroid.x if hasattr(poi.geometry, 'centroid') else lon
                    }
                    amenities.append(amenity_info)
            
            # Cache the result
            if ENABLE_CACHE:
                cache_file = self.cache_dir / f"amenities_{cache_key}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(amenities, f)
            
            return amenities
            
        except Exception as e:
            print(f"Error fetching amenities: {e}")
            return []
    
    def enrich_route(self, route: RouteData) -> Dict:
        """Enrich a single route with OSM data"""
        enrichment = {
            'surface_distribution': {},
            'start_amenities': [],
            'end_amenities': []
        }
        
        # Cache key based on route bounds
        if route.coordinates:
            cache_key = f"{route.activity_id}"
            
            # Get surface data
            enrichment['surface_distribution'] = self.get_surface_data(
                route.coordinates,
                cache_key=cache_key
            )
            
            # Get amenities near start and end
            if route.start_point:
                enrichment['start_amenities'] = self.get_nearby_amenities(
                    route.start_point,
                    radius=AMENITY_SEARCH_RADIUS
                )
            
            if route.end_point:
                enrichment['end_amenities'] = self.get_nearby_amenities(
                    route.end_point,
                    radius=AMENITY_SEARCH_RADIUS
                )
        
        return enrichment
    
    def enrich_routes_batch(
        self,
        routes: List[RouteData],
        show_progress: bool = True
    ) -> List[Dict]:
        """Enrich multiple routes with OSM data"""
        enrichments = []
        
        iterator = tqdm(routes, desc="Enriching routes with OSM data") if show_progress else routes
        
        for route in iterator:
            enrichment = self.enrich_route(route)
            enrichments.append(enrichment)
        
        return enrichments
    
    def get_dominant_surface(self, surface_distribution: Dict[str, float]) -> str:
        """Get the dominant surface type from distribution"""
        if not surface_distribution:
            return "unknown"
        
        # Filter out unknown
        known_surfaces = {k: v for k, v in surface_distribution.items() if k != 'unknown'}
        
        if not known_surfaces:
            return "unknown"
        
        return max(known_surfaces, key=known_surfaces.get)
    
    def has_amenity_type(
        self,
        amenities: List[Dict],
        amenity_type: str
    ) -> bool:
        """Check if amenity list contains a specific type"""
        return any(a['type'] == amenity_type for a in amenities)


if __name__ == "__main__":
    # Test OSM enricher
    from data_processor import ActivityParser
    
    print("Testing OSM Enricher\n")
    
    # Parse a few activities
    parser = ActivityParser()
    routes = parser.parse_all_activities(limit=3)
    
    # Initialize enricher
    enricher = OSMEnricher()
    
    # Enrich routes
    enrichments = enricher.enrich_routes_batch(routes)
    
    # Display results
    for route, enrichment in zip(routes, enrichments):
        print(f"\n{'='*60}")
        print(f"Route: {route.name}")
        print(f"Distance: {route.distance_km:.2f} km")
        print(f"\nSurface Distribution:")
        for surface, percentage in enrichment['surface_distribution'].items():
            print(f"  {surface}: {percentage*100:.1f}%")
        
        dominant = enricher.get_dominant_surface(enrichment['surface_distribution'])
        print(f"Dominant surface: {dominant}")
        
        print(f"\nStart amenities ({len(enrichment['start_amenities'])}):")
        for amenity in enrichment['start_amenities'][:5]:
            print(f"  - {amenity['name']} ({amenity['type']})")
        
        print(f"\nEnd amenities ({len(enrichment['end_amenities'])}):")
        for amenity in enrichment['end_amenities'][:5]:
            print(f"  - {amenity['name']} ({amenity['type']})")
