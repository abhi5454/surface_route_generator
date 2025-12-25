"""
Geocoder module for reverse geocoding coordinates to readable place names.
Uses Geopy with Nominatim (OpenStreetMap) service.
"""
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time

from config import GEOCODE_CACHE_DIR, NOMINATIM_USER_AGENT


class LocationGeocoder:
    """Reverse geocoding service for converting coordinates to place names"""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = Path(cache_dir or GEOCODE_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Nominatim geocoder
        self.geolocator = Nominatim(
            user_agent=NOMINATIM_USER_AGENT,
            timeout=10
        )
        
        # Load cache
        self.cache_file = self.cache_dir / "geocode_cache.pkl"
        self.cache = self._load_cache()
        
        print(f"Geocoder initialized with {len(self.cache)} cached locations")
    
    def _load_cache(self) -> Dict:
        """Load geocode cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_cache(self):
        """Save geocode cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Warning: Failed to save geocode cache: {e}")
    
    def _make_cache_key(self, lat: float, lon: float) -> str:
        """Create cache key from coordinates (rounded to reduce duplicates)"""
        return f"{lat:.5f}_{lon:.5f}"
    
    def get_place_name(
        self,
        lat: float,
        lon: float,
        include_address: bool = False
    ) -> Optional[str]:
        """
        Get readable place name from coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            include_address: If True, return full address; otherwise just area name
            
        Returns:
            Human-readable location name or None if lookup failed
        """
        cache_key = self._make_cache_key(lat, lon)
        
        # Check cache first
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            return cached.get('full' if include_address else 'short')
        
        try:
            # Rate limit to respect Nominatim ToS (1 request per second)
            time.sleep(1.1)
            
            location = self.geolocator.reverse(
                (lat, lon),
                exactly_one=True,
                language='en'
            )
            
            if location:
                address = location.raw.get('address', {})
                
                # Build short location name (neighborhood/suburb + city)
                short_parts = []
                for key in ['neighbourhood', 'suburb', 'village', 'town', 'city_district']:
                    if key in address:
                        short_parts.append(address[key])
                        break
                
                for key in ['city', 'town', 'municipality', 'county']:
                    if key in address:
                        short_parts.append(address[key])
                        break
                
                short_name = ", ".join(short_parts) if short_parts else "Unknown Location"
                full_name = location.address
                
                # Cache the result
                self.cache[cache_key] = {
                    'short': short_name,
                    'full': full_name
                }
                self._save_cache()
                
                return full_name if include_address else short_name
            
            return None
            
        except GeocoderTimedOut:
            print(f"Geocoding timeout for ({lat}, {lon})")
            return None
        except GeocoderServiceError as e:
            print(f"Geocoding error for ({lat}, {lon}): {e}")
            return None
        except Exception as e:
            print(f"Unexpected geocoding error: {e}")
            return None
    
    def get_short_location(self, lat: float, lon: float) -> str:
        """Get short location name (e.g., 'Koramangala, Bangalore')"""
        result = self.get_place_name(lat, lon, include_address=False)
        return result or "Unknown Location"
    
    def get_full_address(self, lat: float, lon: float) -> str:
        """Get full address string"""
        result = self.get_place_name(lat, lon, include_address=True)
        return result or "Unknown Address"
    
    def batch_geocode(
        self,
        coordinates: list[Tuple[float, float]],
        short_form: bool = True
    ) -> list[str]:
        """
        Geocode multiple coordinates.
        
        Args:
            coordinates: List of (lat, lon) tuples
            short_form: If True, return short location names
            
        Returns:
            List of place names
        """
        results = []
        for lat, lon in coordinates:
            if short_form:
                results.append(self.get_short_location(lat, lon))
            else:
                results.append(self.get_full_address(lat, lon))
        return results


if __name__ == "__main__":
    # Test geocoder
    print("Testing Location Geocoder\n")
    
    geocoder = LocationGeocoder()
    
    # Test coordinates (example: Bangalore, India)
    test_coords = [
        (12.9716, 77.5946),  # Bangalore city center
        (12.9352, 77.6245),  # Koramangala
    ]
    
    for lat, lon in test_coords:
        print(f"\nCoordinates: ({lat}, {lon})")
        short = geocoder.get_short_location(lat, lon)
        print(f"  Short: {short}")
        full = geocoder.get_full_address(lat, lon)
        print(f"  Full: {full}")
