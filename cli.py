
"""
Command-line interface for the Surface-Aware Route Generator.
"""
import argparse
import sys
import webbrowser
from typing import Optional, List

from pathlib import Path

# Force UTF-8 output for Windows terminals to handle emojis
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from data_processor import ActivityParser
from embedding_generator import EmbeddingGenerator
from vector_store import VectorStore
from osm_enricher import OSMEnricher
from query_processor import QueryProcessor
from route_recommender import RouteRecommender
from geocoder import LocationGeocoder
from map_visualizer import RouteMapVisualizer
from config import MAX_ACTIVITIES, GOOGLE_API_KEY


class CLI:
    """Command-line interface for route search"""
    
    def __init__(self):
        self.parser = ActivityParser()
        self.generator = EmbeddingGenerator()
        self.store = VectorStore()
        self.osm_enricher = OSMEnricher()
        
        # Only initialize query processor if API key is available
        if GOOGLE_API_KEY:
            self.query_processor = QueryProcessor()
            self.use_gemini = True
        else:
            self.query_processor = None
            self.use_gemini = False
            print("Note: Gemini query processing disabled (no API key)")
        
        self.recommender = RouteRecommender(
            vector_store=self.store,
            embedding_generator=self.generator,
            osm_enricher=self.osm_enricher,
            query_processor=self.query_processor
        )
        
        # Initialize geocoder and map visualizer
        self.geocoder = LocationGeocoder()
        self.map_visualizer = RouteMapVisualizer()
    
    def initialize_database(self, limit: int = None, skip_osm: bool = False):
        """Initialize the database with activity data"""
        print(f"\n{'='*70}")
        print("INITIALIZING DATABASE")
        print(f"{'='*70}\n")
        
        # Check if database already has data
        if self.store.count() > 0:
            response = input(f"Database already contains {self.store.count()} routes. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("Keeping existing database.")
                return
            self.store.clear()
        
        # Parse activities
        print("Step 1: Parsing activity files...")
        routes = self.parser.parse_all_activities(limit=limit or MAX_ACTIVITIES)
        
        if not routes:
            print("No routes found!")
            return
        
        print(f"Parsed {len(routes)} routes")
        
        # Generate embeddings
        print("\nStep 2: Generating embeddings...")
        embeddings = self.generator.generate_route_embeddings(routes)
        
        # Enrich with OSM data (optional - can be slow)
        enrichments = None
        if not skip_osm:
            response = input("\nEnrich with OSM data? This may take several minutes. (y/N): ")
            if response.lower() == 'y':
                print("\nStep 3: Enriching with OSM data...")
                enrichments = self.osm_enricher.enrich_routes_batch(routes)
        
        # Store in vector database
        print("\nStep 4: Storing in vector database...")
        self.store.add_routes(routes, embeddings, metadata_extra=enrichments)
        
        print(f"\n{'='*70}")
        print(f"✓ Database initialized with {self.store.count()} routes")
        print(f"{'='*70}\n")
    
    def search_interactive(self):
        """Interactive search mode"""
        print(f"\n{'='*70}")
        print("SURFACE-AWARE ROUTE GENERATOR")
        print(f"{'='*70}")
        print(f"Database: {self.store.count()} routes")
        print(f"Gemini AI: {'Enabled' if self.use_gemini else 'Disabled'}")
        print(f"{'='*70}\n")
        
        if self.store.count() == 0:
            print("Database is empty! Run with --init first.")
            return
        
        print("Enter your route search queries (or 'quit' to exit)")
        print("Example: 'Find a 5km run with moderate elevation'\n")
        
        while True:
            try:
                query = input("\n> ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not query:
                    continue
                
                # Search for routes
                recommendations = self.recommender.search_routes(
                    query,
                    n_results=5,
                    use_llm=self.use_gemini
                )
                
                # Display results
                print(self.recommender.format_recommendations(recommendations))
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def search_single(
        self,
        query: str,
        n_results: int = 5,
        show_locations: bool = True,
        show_map: bool = False,
        export_map: str = None
    ):
        """Single search query with visualization options"""
        if self.store.count() == 0:
            print("Database is empty! Run with --init first.")
            return
        
        print(f"\nSearching for: {query}\n")
        
        recommendations = self.recommender.search_routes(
            query,
            n_results=n_results,
            use_llm=self.use_gemini
        )
        
        # Geocode locations if requested
        if show_locations and recommendations:
            print("Geocoding locations...")
            self._add_location_names(recommendations)
        
        # Display results
        print(self.recommender.format_recommendations(recommendations, show_details=True))
        
        # Show map if requested
        if (show_map or export_map) and recommendations:
            self._visualize_routes(recommendations, show_map, export_map)
    
    def _add_location_names(self, recommendations):
        """Add location names to recommendations via reverse geocoding"""
        for rec in recommendations:
            if rec.start_coords:
                rec.start_location = self.geocoder.get_short_location(
                    rec.start_coords[0], rec.start_coords[1]
                )
            if rec.end_coords:
                rec.end_location = self.geocoder.get_short_location(
                    rec.end_coords[0], rec.end_coords[1]
                )
    
    def _visualize_routes(self, recommendations, show_in_browser: bool, export_path: str = None):
        """Create and display route map"""
        if not recommendations:
            return
        
        # For now, visualize the top recommendation
        top_rec = recommendations[0]
        
        # Try to get full coordinates from file
        coords = []
        if top_rec.metadata and top_rec.metadata.get('file_path'):
            file_path = Path(top_rec.metadata.get('file_path'))
            try:
                # We need to handle potential relative paths or just use the name if in expected dir
                # Only re-parse if file exists
                if not file_path.exists():
                    # Try finding it in ACTIVITIES_DIR if it was stored as just name or relative
                    from config import ACTIVITIES_DIR
                    potential_path = ACTIVITIES_DIR / file_path.name
                    if potential_path.exists():
                        file_path = potential_path
                
                if file_path.exists():
                    print(f"Loading full route from: {file_path.name}")
                    route_data = self.parser.parse_activity_file(file_path)
                    if route_data:
                        coords = route_data.coordinates
            except Exception as e:
                print(f"Error loading full route path: {e}")
        
        # Fallback to simple line if file parsing failed
        if not coords:
            if not top_rec.start_coords or not top_rec.end_coords:
                print("Cannot visualize: No coordinate data available")
                return
            print("Warning: Could not load full route data, showing straight line start-to-end.")
            coords = [top_rec.start_coords, top_rec.end_coords]
        
        route_map = self.map_visualizer.create_route_map(
            coordinates=coords,
            route_name=top_rec.route_name,
            distance_km=top_rec.distance_km,
            elevation_gain=top_rec.elevation_gain,
            start_location=top_rec.start_location,
            end_location=top_rec.end_location
        )
        
        if export_path:
            route_map.save(export_path)
            print(f"\nMap exported to: {export_path}")
        
        if show_in_browser:
            save_path = self.map_visualizer.open_in_browser(
                route_map,
                f"route_{top_rec.route_id}.html"
            )
            print(f"\nMap opened in browser: {save_path}")


def main():
    """Main entry point"""
    arg_parser = argparse.ArgumentParser(
        description="Surface-Aware Route Generator - Find running/cycling routes using natural language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Initialize database with all activities:
    python cli.py --init
  
  Initialize with first 50 activities (faster):
    python cli.py --init --limit 50
  
  Interactive search mode:
    python cli.py
  
  Single query:
    python cli.py --query "Find a 10km loop with a cafe at the end"
        """
    )
    
    arg_parser.add_argument(
        '--init',
        action='store_true',
        help='Initialize database with activity data'
    )
    
    arg_parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of activities to process'
    )
    
    arg_parser.add_argument(
        '--skip-osm',
        action='store_true',
        help='Skip OSM enrichment (faster initialization)'
    )
    
    arg_parser.add_argument(
        '--query',
        type=str,
        help='Single search query'
    )
    
    arg_parser.add_argument(
        '--num-results',
        type=int,
        default=5,
        help='Number of results to return (default: 5)'
    )
    
    arg_parser.add_argument(
        '--show-locations',
        action='store_true',
        default=True,
        help='Show place names for start/end points (default: enabled)'
    )
    
    arg_parser.add_argument(
        '--no-locations',
        action='store_true',
        help='Disable location geocoding (faster)'
    )
    
    arg_parser.add_argument(
        '--show-map',
        action='store_true',
        help='Open route map in browser'
    )
    
    arg_parser.add_argument(
        '--export-map',
        type=str,
        metavar='FILE',
        help='Export route map to HTML file'
    )
    
    args = arg_parser.parse_args()
    
    # Initialize CLI
    try:
        cli = CLI()
    except Exception as e:
        print(f"Error initializing CLI: {e}")
        return 1
    
    # Handle initialization
    if args.init:
        cli.initialize_database(limit=args.limit, skip_osm=args.skip_osm)
        return 0
    
    # Handle single query
    if args.query:
        cli.search_single(
            args.query,
            n_results=args.num_results,
            show_locations=not args.no_locations,
            show_map=args.show_map,
            export_map=args.export_map
        )
        return 0
    
    # Default: interactive mode
    cli.search_interactive()
    return 0


if __name__ == "__main__":
    sys.exit(main())
