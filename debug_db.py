"""Test the fixed search_by_filters"""
from vector_store import VectorStore
from embedding_generator import EmbeddingGenerator

store = VectorStore()
gen = EmbeddingGenerator()

query = "5km run"
query_emb = gen.generate_single_embedding(query)

print("\n--- Testing FIXED search_by_filters ---")
results = store.search_by_filters(
    query_emb, 
    distance_range=(4.0, 6.0),
    n_results=3
)

print(f"IDs: {results.get('ids')}")
print(f"Metadatas: {results.get('metadatas')}")
print(f"Distances: {results.get('distances')}")

# Test the full flow
if results.get('ids') and results['ids'][0]:
    print("\n--- Extracted Values ---")
    for i, route_id in enumerate(results['ids'][0]):
        meta = results['metadatas'][0][i] if i < len(results.get('metadatas', [[]])[0]) else {}
        print(f"Route {route_id}:")
        print(f"  Name: {meta.get('name', 'Unknown')}")
        print(f"  Distance: {meta.get('distance_km', '0')} km")
