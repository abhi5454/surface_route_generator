"""Surface-Aware Route Generator package"""

__version__ = "0.1.0"
__author__ = "Your Name"
__description__ = "Gen AI-powered route discovery using Strava data and OpenStreetMap"

from .data_processor import ActivityParser, RouteData
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore
from .osm_enricher import OSMEnricher
from .query_processor import QueryProcessor
from .route_recommender import RouteRecommender

__all__ = [
    'ActivityParser',
    'RouteData',
    'EmbeddingGenerator',
    'VectorStore',
    'OSMEnricher',
    'QueryProcessor',
    'RouteRecommender',
]
