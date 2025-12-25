"""
Main entry point for the Surface-Aware Route Generator.
Orchestrates the data processing pipeline.
"""
from pathlib import Path
import argparse

from data_processor import ActivityParser
from embedding_generator import EmbeddingGenerator
from vector_store import VectorStore
from osm_enricher import OSMEnricher
from config import MAX_ACTIVITIES


def build_database(limit: int = None, skip_osm: bool = False):
    """Build the route database from scratch"""
    print("="*70)
    print("BUILDING ROUTE DATABASE")
    print("="*70)
    
    # Step 1: Parse activities
    print("\n[1/4] Parsing activity files...")
    parser = ActivityParser()
    routes = parser.parse_all_activities(limit=limit or MAX_ACTIVITIES)
    
    if not routes:
        print("No routes found!")
        return
    
    print(f"Successfully parsed {len(routes)} routes")
    
    # Step 2: Generate embeddings
    print("\n[2/4] Generating embeddings with PyTorch...")
    generator = EmbeddingGenerator()
    embeddings = generator.generate_route_embeddings(routes)
    
    # Step 3: OSM enrichment (optional)
    enrichments = None
    if not skip_osm:
        print("\n[3/4] Enriching with OSM data...")
        enricher = OSMEnricher()
        enrichments = enricher.enrich_routes_batch(routes, show_progress=True)
    else:
        print("\n[3/4] Skipping OSM enrichment...")
    
    # Step 4: Build vector database
    print("\n[4/4] Building vector database...")
    store = VectorStore()
    
    # Clear existing data
    if store.count() > 0:
        print(f"Clearing {store.count()} existing routes...")
        store.clear()
    
    # Add routes
    store.add_routes(routes, embeddings, metadata_extra=enrichments)
    
    print("\n" + "="*70)
    print(f"✓ DATABASE BUILD COMPLETE")
    print(f"✓ Total routes: {store.count()}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Build the Surface-Aware Route Generator database")
    parser.add_argument('--limit', type=int, help='Limit number of activities to process')
    parser.add_argument('--skip-osm', action='store_true', help='Skip OSM enrichment')
    
    args = parser.parse_args()
    
    build_database(limit=args.limit, skip_osm=args.skip_osm)


if __name__ == "__main__":
    main()
