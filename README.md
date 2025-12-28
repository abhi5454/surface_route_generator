# Surface-Aware Route Generator

Gen AI-powered route discovery using Strava data, OpenStreetMap, and semantic search.

## Features

- **Natural Language Queries**: "Find me a 10km loop with a coffee shop at the end"
- **Google Gemini Integration**: Intelligent query understanding
- **Semantic Search**: Vector embeddings with PyTorch
- **OSM Integration**: Surface types and amenities
- **GPU Acceleration**: PyTorch with CUDA support

## Setup

### 1. Install Dependencies

```bash
# Install PyTorch with CUDA support (if you have an NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file in this directory:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get your API key from: https://makersuite.google.com/app/apikey

### 3. Initialize Database

```bash
# Process all activities (may take 10-30 minutes)
python main.py

# Or process just a subset for testing
python main.py --limit 50

# Skip OSM enrichment for faster processing
python main.py --skip-osm
```

## Usage

### Interactive Mode

```bash
python cli.py
```

Then enter queries like:
- "Find a 5km run with moderate elevation"
- "Show me a 10km loop with mostly paved roads"
- "Short walk near a park"

### Single Query

```bash
python cli.py --query "15km cycling route with a cafe"
```

### Options

```bash
python cli.py --help
```

## Architecture

- **data_processor.py**: Parse GPX/FIT files
- **embedding_generator.py**: Generate route embeddings with sentence-transformers
- **vector_store.py**: ChromaDB for similarity search
- **osm_enricher.py**: Fetch surface & amenity data from OpenStreetMap
- **query_processor.py**: Natural language understanding with Gemini
- **route_recommender.py**: Recommendation engine
- **cli.py**: Command-line interface

## Configuration

Edit `config.py` to customize:
- Embedding model
- Search parameters
- OSM amenity types
- Distance tolerances

## Requirements

- Python 3.8+
- NVIDIA GPU (recommended for faster embeddings)
- Internet connection (for OSM queries)
- Google API key (for natural language queries with Gemini)
