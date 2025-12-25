"""
Data processor for parsing GPX and FIT files from Strava activities.
"""
import gzip
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gpxpy
import pandas as pd
from fitparse import FitFile
from tqdm import tqdm
import numpy as np

from config import ACTIVITIES_DIR, MIN_ROUTE_DISTANCE, MAX_ROUTE_DISTANCE


class RouteData:
    """Container for route information"""
    
    def __init__(
        self,
        activity_id: str,
        name: str,
        activity_type: str,
        coordinates: List[Tuple[float, float]],
        elevations: List[float],
        timestamps: List,
        distance_km: float,
        elevation_gain: float,
        elevation_loss: float,
        file_path: Path
    ):
        self.activity_id = activity_id
        self.name = name
        self.activity_type = activity_type
        self.coordinates = coordinates  # [(lat, lon), ...]
        self.elevations = elevations
        self.timestamps = timestamps
        self.distance_km = distance_km
        self.elevation_gain = elevation_gain
        self.elevation_loss = elevation_loss
        self.file_path = file_path
        
        # Calculate derived properties
        self.start_point = coordinates[0] if coordinates else None
        self.end_point = coordinates[-1] if coordinates else None
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            "activity_id": self.activity_id,
            "name": self.name,
            "activity_type": self.activity_type,
            "distance_km": self.distance_km,
            "elevation_gain": self.elevation_gain,
            "elevation_loss": self.elevation_loss,
            "start_lat": self.start_point[0] if self.start_point else None,
            "start_lon": self.start_point[1] if self.start_point else None,
            "end_lat": self.end_point[0] if self.end_point else None,
            "end_lon": self.end_point[1] if self.end_point else None,
            "num_points": len(self.coordinates),
            "file_path": str(self.file_path)
        }
    
    def get_description(self) -> str:
        """Generate text description for embedding"""
        desc_parts = [
            f"{self.name or 'Untitled activity'}",
            f"Type: {self.activity_type}",
            f"Distance: {self.distance_km:.2f} km",
            f"Elevation gain: {self.elevation_gain:.0f}m"
        ]
        
        if self.elevation_loss > 0:
            desc_parts.append(f"Elevation loss: {self.elevation_loss:.0f}m")
            
        return ". ".join(desc_parts)


class ActivityParser:
    """Parser for GPX and FIT activity files"""
    
    def __init__(self, activities_dir: Path = ACTIVITIES_DIR):
        self.activities_dir = Path(activities_dir)
        
    def parse_gpx(self, file_path: Path) -> Optional[RouteData]:
        """Parse a GPX file"""
        try:
            # Handle gzipped files
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'r') as f:
                    gpx = gpxpy.parse(f)
            else:
                with open(file_path, 'r') as f:
                    gpx = gpxpy.parse(f)
            
            coordinates = []
            elevations = []
            timestamps = []
            
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        coordinates.append((point.latitude, point.longitude))
                        elevations.append(point.elevation if point.elevation else 0.0)
                        timestamps.append(point.time)
            
            if not coordinates:
                return None
            
            # Calculate distance and elevation
            distance_km = self._calculate_distance(coordinates)
            elevation_gain, elevation_loss = self._calculate_elevation_change(elevations)
            
            # Filter by distance
            if distance_km < MIN_ROUTE_DISTANCE or distance_km > MAX_ROUTE_DISTANCE:
                return None
            
            # Extract metadata
            activity_id = file_path.stem.split('.')[0]
            name = gpx.tracks[0].name if gpx.tracks and gpx.tracks[0].name else "Untitled"
            activity_type = gpx.tracks[0].type if gpx.tracks and gpx.tracks[0].type else "Unknown"
            
            return RouteData(
                activity_id=activity_id,
                name=name,
                activity_type=activity_type,
                coordinates=coordinates,
                elevations=elevations,
                timestamps=timestamps,
                distance_km=distance_km,
                elevation_gain=elevation_gain,
                elevation_loss=elevation_loss,
                file_path=file_path
            )
            
        except Exception as e:
            print(f"Error parsing GPX file {file_path}: {e}")
            return None
    
    def parse_fit(self, file_path: Path) -> Optional[RouteData]:
        """Parse a FIT file"""
        try:
            # Handle gzipped files - need to read into memory first
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rb') as f:
                    fit_data = io.BytesIO(f.read())
                fitfile = FitFile(fit_data)
            else:
                fitfile = FitFile(str(file_path))
            
            coordinates = []
            elevations = []
            timestamps = []
            
            # Extract session info for metadata
            name = "Untitled"
            activity_type = "Unknown"
            
            for record in fitfile.get_messages('session'):
                for field in record:
                    if field.name == 'sport':
                        activity_type = field.value
                        
            # Extract track points
            for record in fitfile.get_messages('record'):
                lat, lon, ele, timestamp = None, None, None, None
                
                for field in record:
                    if field.name == 'position_lat':
                        lat = field.value * (180 / 2**31) if field.value else None
                    elif field.name == 'position_long':
                        lon = field.value * (180 / 2**31) if field.value else None
                    elif field.name == 'altitude' or field.name == 'enhanced_altitude':
                        ele = field.value
                    elif field.name == 'timestamp':
                        timestamp = field.value
                
                if lat is not None and lon is not None:
                    coordinates.append((lat, lon))
                    elevations.append(ele if ele else 0.0)
                    timestamps.append(timestamp)
            
            if not coordinates:
                return None
            
            # Calculate distance and elevation
            distance_km = self._calculate_distance(coordinates)
            elevation_gain, elevation_loss = self._calculate_elevation_change(elevations)
            
            # Filter by distance
            if distance_km < MIN_ROUTE_DISTANCE or distance_km > MAX_ROUTE_DISTANCE:
                return None
            
            activity_id = file_path.stem.split('.')[0]
            
            return RouteData(
                activity_id=activity_id,
                name=name,
                activity_type=activity_type,
                coordinates=coordinates,
                elevations=elevations,
                timestamps=timestamps,
                distance_km=distance_km,
                elevation_gain=elevation_gain,
                elevation_loss=elevation_loss,
                file_path=file_path
            )
            
        except Exception as e:
            print(f"Error parsing FIT file {file_path}: {e}")
            return None
    
    def parse_activity_file(self, file_path: Path) -> Optional[RouteData]:
        """Parse an activity file (auto-detect format)"""
        if '.gpx' in file_path.suffixes:
            return self.parse_gpx(file_path)
        elif '.fit' in file_path.suffixes:
            return self.parse_fit(file_path)
        else:
            return None
    
    def parse_all_activities(self, limit: Optional[int] = None) -> List[RouteData]:
        """Parse all activity files in the directory"""
        activity_files = []
        
        # Find all GPX and FIT files
        for pattern in ['*.gpx', '*.gpx.gz', '*.fit', '*.fit.gz']:
            activity_files.extend(self.activities_dir.glob(pattern))
        
        print(f"Found {len(activity_files)} activity files")
        
        if limit:
            activity_files = activity_files[:limit]
            print(f"Limiting to {limit} files")
        
        routes = []
        for file_path in tqdm(activity_files, desc="Parsing activities"):
            route = self.parse_activity_file(file_path)
            if route:
                routes.append(route)
        
        print(f"Successfully parsed {len(routes)} routes")
        return routes
    
    @staticmethod
    def _calculate_distance(coordinates: List[Tuple[float, float]]) -> float:
        """Calculate total distance in kilometers using Haversine formula"""
        if len(coordinates) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(coordinates) - 1):
            lat1, lon1 = coordinates[i]
            lat2, lon2 = coordinates[i + 1]
            
            # Haversine formula
            R = 6371  # Earth radius in km
            
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            
            a = (np.sin(dlat / 2) ** 2 +
                 np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
                 np.sin(dlon / 2) ** 2)
            
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            total_distance += R * c
        
        return total_distance
    
    @staticmethod
    def _calculate_elevation_change(elevations: List[float]) -> Tuple[float, float]:
        """Calculate cumulative elevation gain and loss"""
        if len(elevations) < 2:
            return 0.0, 0.0
        
        gain = 0.0
        loss = 0.0
        
        for i in range(len(elevations) - 1):
            diff = elevations[i + 1] - elevations[i]
            if diff > 0:
                gain += diff
            else:
                loss += abs(diff)
        
        return gain, loss


def load_activities_metadata() -> pd.DataFrame:
    """Load activities.csv for additional metadata"""
    csv_path = ACTIVITIES_DIR.parent / "activities.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


if __name__ == "__main__":
    # Test the parser
    parser = ActivityParser()
    routes = parser.parse_all_activities(limit=10)
    
    print(f"\n{'='*60}")
    print(f"Parsed {len(routes)} routes")
    print(f"{'='*60}\n")
    
    for route in routes[:5]:
        print(f"Activity: {route.name}")
        print(f"Type: {route.activity_type}")
        print(f"Distance: {route.distance_km:.2f} km")
        print(f"Elevation: +{route.elevation_gain:.0f}m / -{route.elevation_loss:.0f}m")
        print(f"Points: {len(route.coordinates)}")
        print(f"Description: {route.get_description()}")
        print(f"{'-'*60}\n")
