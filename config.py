"""
Configuration module for Surface-Aware Route Generator.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
ACTIVITIES_DIR = BASE_DIR / "activities"
DATA_DIR = Path(__file__).parent / "data"
DB_DIR = DATA_DIR / "vector_db"
CACHE_DIR = DATA_DIR / "cache"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ============================================================================
# API Keys
# ============================================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # For Gemini

# ============================================================================
# LLM Configuration
# ============================================================================
GEMINI_MODEL = "gemini-2.5-flash-lite"  # Available for this key
GEMINI_MAX_TOKENS = 4096

# ============================================================================
# PyTorch Configuration
# ============================================================================
# Automatically select best available device (CUDA > MPS > CPU)
if torch.cuda.is_available():
    DEVICE = "cuda"
    TORCH_DTYPE = torch.float16  # Use fp16 for GPU
elif torch.backends.mps.is_available():
    DEVICE = "mps"  # Apple Silicon
    TORCH_DTYPE = torch.float32
else:
    DEVICE = "cpu"
    TORCH_DTYPE = torch.float32

print(f"Using device: {DEVICE}")

# ============================================================================
# Embedding Configuration
# ============================================================================
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_MAX_LENGTH = 512

# ============================================================================
# Vector Database Configuration
# ============================================================================
CHROMA_COLLECTION_NAME = "strava_routes"
CHROMA_PERSIST_DIR = str(DB_DIR)

# ============================================================================
# OSM Configuration
# ============================================================================
OSM_CACHE_DIR = CACHE_DIR / "osm_cache"
OSM_CACHE_DIR.mkdir(exist_ok=True)

# Amenity search radius in meters
AMENITY_SEARCH_RADIUS = 100

# Default location (Bangalore coordinates)
DEFAULT_LOCATION = (12.9716, 77.5946)  # Lat, Lon

# Surface types to track
SURFACE_TYPES = [
    "paved", "asphalt", "concrete",
    "unpaved", "gravel", "dirt", "ground",
    "grass", "sand", "wood"
]

# Amenities to search for
AMENITY_TYPES = [
    "cafe", "restaurant", "bar", "pub",
    "fast_food", "food_court", "ice_cream",
    "park", "viewpoint", "bench"
]

# ============================================================================
# Route Processing Configuration
# ============================================================================
# Maximum number of activities to process (None for all)
MAX_ACTIVITIES = None

# Minimum route distance in km
MIN_ROUTE_DISTANCE = 0.5

# Maximum route distance in km  
MAX_ROUTE_DISTANCE = 100

# Distance tolerance for matching queries (±%)
DISTANCE_TOLERANCE = 0.2  # 20%

# ============================================================================
# Search Configuration
# ============================================================================
# Number of results to return
DEFAULT_NUM_RESULTS = 5

# Minimum similarity score (0-1)
MIN_SIMILARITY_SCORE = 0.5

# ============================================================================
# Cache Configuration
# ============================================================================
ENABLE_CACHE = True
CACHE_EXPIRY_DAYS = 30

# ============================================================================
# Geocoding Configuration
# ============================================================================
GEOCODE_CACHE_DIR = CACHE_DIR / "geocode_cache"
GEOCODE_CACHE_DIR.mkdir(exist_ok=True)

# Nominatim requires a unique user agent (OSM policy)
NOMINATIM_USER_AGENT = "strava-route-generator-1.0"

# ============================================================================
# Map Visualization Configuration
# ============================================================================
MAPS_OUTPUT_DIR = DATA_DIR / "maps"
MAPS_OUTPUT_DIR.mkdir(exist_ok=True)

# Map tile providers
MAP_TILES = "OpenStreetMap"  # Options: OpenStreetMap, CartoDB positron, etc.

# Default map zoom level
DEFAULT_MAP_ZOOM = 13

