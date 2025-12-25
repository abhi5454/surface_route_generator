"""
Query processor using Google Gemini for natural language understanding.
"""
import google.genai as genai
from typing import Dict, List, Optional, Tuple
import json
import re

from config import GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_MAX_TOKENS


class QueryProcessor:
    """Process natural language queries using Google Gemini"""
    
    def __init__(self):
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not set in environment")
        
        # Configure Gemini
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        
        print(f"Query Processor initialized with model: {GEMINI_MODEL}")
    
    def parse_query(self, user_query: str) -> Dict:
        """Parse natural language query into structured parameters"""
        
        prompt = """You are a route search assistant. Parse user queries about running/cycling routes into structured JSON parameters.

Extract the following information:
1. distance_km: Target distance in kilometers (extract number)
2. distance_tolerance: Acceptable variance (default 0.2 for ±20%)
3. route_type: Type of route (loop, out-and-back, point-to-point)
4. surface_preferences: List of preferred surfaces (paved, unpaved, gravel, dirt, etc.)
5. amenities_required: List of amenities needed (cafe, restaurant, park, etc.)
6. location: Specific location or area mentioned
7. elevation_preference: low, moderate, high, or null
8. activity_type: Run, Ride, Walk, Hike, etc.

Return ONLY valid JSON with these fields. Use null for missing information.
Distance should always be in kilometers.

Parse this route search query:

"{query}"

Return structured JSON parameters."""

        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt.format(query=user_query),
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=GEMINI_MAX_TOKENS,
                    temperature=0.1,
                )
            )
            
            # Extract JSON from response
            content = response.text
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = json.loads(content)
            
            return self._validate_and_normalize(parsed)
            
        except Exception as e:
            print(f"Error parsing query with Gemini: {e}")
            # Fallback to simple parsing
            return self._fallback_parse(user_query)
    
    def _validate_and_normalize(self, parsed: Dict) -> Dict:
        """Validate and normalize parsed parameters"""
        normalized = {
            'distance_km': None,
            'distance_range': None,
            'distance_tolerance': 0.2,
            'route_type': None,
            'surface_preferences': [],
            'amenities_required': [],
            'location': None,
            'elevation_preference': None,
            'activity_type': None
        }
        
        # Extract and validate distance
        if parsed.get('distance_km'):
            dist = float(parsed['distance_km'])
            tolerance = float(parsed.get('distance_tolerance', 0.2))
            normalized['distance_km'] = dist
            normalized['distance_tolerance'] = tolerance
            normalized['distance_range'] = (
                dist * (1 - tolerance),
                dist * (1 + tolerance)
            )
        
        # Copy other fields
        for key in ['route_type', 'location', 'elevation_preference', 'activity_type']:
            if parsed.get(key):
                normalized[key] = parsed[key]
        
        # Handle lists
        if parsed.get('surface_preferences'):
            normalized['surface_preferences'] = (
                parsed['surface_preferences'] if isinstance(parsed['surface_preferences'], list)
                else [parsed['surface_preferences']]
            )
        
        if parsed.get('amenities_required'):
            normalized['amenities_required'] = (
                parsed['amenities_required'] if isinstance(parsed['amenities_required'], list)
                else [parsed['amenities_required']]
            )
        
        return normalized
    
    def _fallback_parse(self, query: str) -> Dict:
        """Simple fallback parser if Gemini fails"""
        query_lower = query.lower()
        
        params = {
            'distance_km': None,
            'distance_range': None,
            'distance_tolerance': 0.2,
            'route_type': None,
            'surface_preferences': [],
            'amenities_required': [],
            'location': None,
            'elevation_preference': None,
            'activity_type': None
        }
        
        # Extract distance
        dist_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:km|kilometer)', query_lower)
        if dist_match:
            dist = float(dist_match.group(1))
            params['distance_km'] = dist
            params['distance_range'] = (dist * 0.8, dist * 1.2)
        
        # Detect route type
        if 'loop' in query_lower:
            params['route_type'] = 'loop'
        elif 'out and back' in query_lower or 'out-and-back' in query_lower:
            params['route_type'] = 'out-and-back'
        
        # Detect amenities
        if 'cafe' in query_lower or 'coffee' in query_lower:
            params['amenities_required'].append('cafe')
        if 'restaurant' in query_lower:
            params['amenities_required'].append('restaurant')
        if 'park' in query_lower:
            params['amenities_required'].append('park')
        
        # Detect surface
        if 'paved' in query_lower:
            params['surface_preferences'].append('paved')
        if 'unpaved' in query_lower or 'trail' in query_lower:
            params['surface_preferences'].append('unpaved')
        
        # Detect activity type
        if 'run' in query_lower:
            params['activity_type'] = 'Run'
        elif 'ride' in query_lower or 'bike' in query_lower or 'cycle' in query_lower:
            params['activity_type'] = 'Ride'
        elif 'walk' in query_lower:
            params['activity_type'] = 'Walk'
        
        return params
    
    def generate_explanation(
        self,
        query: str,
        results: List[Dict],
        query_params: Dict
    ) -> str:
        """Generate human-readable explanation of search results"""
        
        if not results:
            return "No routes found matching your criteria."
        
        explanation_parts = [
            f"Found {len(results)} routes matching your search",
        ]
        
        if query_params.get('distance_km'):
            explanation_parts.append(
                f"around {query_params['distance_km']} km"
            )
        
        if query_params.get('amenities_required'):
            amenities_str = ", ".join(query_params['amenities_required'])
            explanation_parts.append(
                f"with nearby {amenities_str}"
            )
        
        return " ".join(explanation_parts) + "."


if __name__ == "__main__":
    # Test query processor
    import os
    
    # Check if API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY not set")
        print("Please set the API key in your .env file to test the query processor")
        print("\nExample .env file:")
        print("GOOGLE_API_KEY=your_api_key_here")
        exit(1)
    
    print("Testing Query Processor\n")
    
    processor = QueryProcessor()
    
    # Test queries
    test_queries = [
        "Find me a 10km loop in Bangalore that avoids heavy traffic and has a coffee shop at the end",
        "5 kilometer run with moderate elevation",
        "Show me a paved cycling route around 15km with a park nearby",
        "Short 3km walk on trails"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print(f"{'='*60}")
        
        params = processor.parse_query(query)
        
        print("Parsed parameters:")
        print(json.dumps(params, indent=2))
