"""
Map visualization module for generating interactive route maps.
Uses Folium to create HTML maps with route polylines and markers.
"""
import folium
from folium import plugins
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import webbrowser

from config import MAPS_OUTPUT_DIR, MAP_TILES, DEFAULT_MAP_ZOOM
from geocoder import LocationGeocoder


class RouteMapVisualizer:
    """Generate interactive maps for route visualization"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = Path(output_dir or MAPS_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.geocoder = LocationGeocoder()
        
        print("Map visualizer initialized")
    
    def create_route_map(
        self,
        coordinates: List[Tuple[float, float]],
        route_name: str = "Route",
        distance_km: float = 0,
        elevation_gain: float = 0,
        start_location: str = None,
        end_location: str = None,
        output_path: str = None
    ) -> folium.Map:
        """
        Create an interactive map for a single route.
        
        Args:
            coordinates: List of (lat, lon) tuples
            route_name: Name to display for the route
            distance_km: Route distance in km
            elevation_gain: Elevation gain in meters
            start_location: Start location name (will geocode if not provided)
            end_location: End location name (will geocode if not provided)
            output_path: Path to save HTML file (optional)
            
        Returns:
            Folium Map object
        """
        if not coordinates:
            raise ValueError("No coordinates provided")
        
        # Calculate map center
        lats = [c[0] for c in coordinates]
        lons = [c[1] for c in coordinates]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        # Create map
        route_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=DEFAULT_MAP_ZOOM,
            tiles=MAP_TILES
        )
        
        # Add route polyline
        folium.PolyLine(
            locations=coordinates,
            weight=4,
            color='#3388ff',
            opacity=0.8,
            popup=f"{route_name}<br>{distance_km:.1f} km"
        ).add_to(route_map)
        
        # Get location names if not provided
        start_point = coordinates[0]
        end_point = coordinates[-1]
        
        if not start_location:
            start_location = self.geocoder.get_short_location(start_point[0], start_point[1])
        if not end_location:
            end_location = self.geocoder.get_short_location(end_point[0], end_point[1])
        
        # Add start marker (green)
        folium.Marker(
            location=start_point,
            popup=folium.Popup(
                f"<b>🚀 Start</b><br>{start_location}<br><br>"
                f"<b>Route:</b> {route_name}<br>"
                f"<b>Distance:</b> {distance_km:.1f} km<br>"
                f"<b>Elevation:</b> +{elevation_gain:.0f}m",
                max_width=300
            ),
            icon=folium.Icon(color='green', icon='play', prefix='fa'),
            tooltip="Start"
        ).add_to(route_map)
        
        # Add end marker (red)
        folium.Marker(
            location=end_point,
            popup=folium.Popup(
                f"<b>🏁 End</b><br>{end_location}",
                max_width=300
            ),
            icon=folium.Icon(color='red', icon='flag', prefix='fa'),
            tooltip="End"
        ).add_to(route_map)
        
        # Fit map bounds to route
        route_map.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(route_map)
        
        # Save if output path provided
        if output_path:
            save_path = Path(output_path)
            route_map.save(str(save_path))
            print(f"Map saved to: {save_path}")
        
        return route_map
    
    def create_multi_route_map(
        self,
        routes: List[Dict],
        output_path: str = None
    ) -> folium.Map:
        """
        Create a map comparing multiple routes.
        
        Args:
            routes: List of route dictionaries with keys:
                - coordinates: List of (lat, lon)
                - name: Route name
                - distance_km: Distance
                - elevation_gain: Elevation
                - color: Optional line color
            output_path: Path to save HTML file
            
        Returns:
            Folium Map object
        """
        if not routes:
            raise ValueError("No routes provided")
        
        # Calculate overall center
        all_lats = []
        all_lons = []
        for route in routes:
            coords = route.get('coordinates', [])
            all_lats.extend([c[0] for c in coords])
            all_lons.extend([c[1] for c in coords])
        
        center_lat = sum(all_lats) / len(all_lats) if all_lats else 0
        center_lon = sum(all_lons) / len(all_lons) if all_lons else 0
        
        # Create map
        multi_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=DEFAULT_MAP_ZOOM,
            tiles=MAP_TILES
        )
        
        # Color palette for routes
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
        
        for i, route in enumerate(routes):
            coords = route.get('coordinates', [])
            name = route.get('name', f'Route {i+1}')
            distance = route.get('distance_km', 0)
            elevation = route.get('elevation_gain', 0)
            color = route.get('color', colors[i % len(colors)])
            
            if not coords:
                continue
            
            # Add route polyline
            folium.PolyLine(
                locations=coords,
                weight=4,
                color=color,
                opacity=0.8,
                popup=f"<b>{name}</b><br>{distance:.1f} km<br>+{elevation:.0f}m"
            ).add_to(multi_map)
            
            # Add start marker
            start = coords[0]
            folium.CircleMarker(
                location=start,
                radius=8,
                color=color,
                fill=True,
                popup=f"Start: {name}"
            ).add_to(multi_map)
        
        # Fit bounds
        if all_lats and all_lons:
            multi_map.fit_bounds([
                [min(all_lats), min(all_lons)],
                [max(all_lats), max(all_lons)]
            ])
        
        # Add fullscreen and layer control
        plugins.Fullscreen().add_to(multi_map)
        
        # Save if output path provided
        if output_path:
            save_path = Path(output_path)
            multi_map.save(str(save_path))
            print(f"Multi-route map saved to: {save_path}")
        
        return multi_map
    
    def open_in_browser(self, map_obj: folium.Map, filename: str = "route_map.html"):
        """Save map and open in default browser"""
        save_path = self.output_dir / filename
        map_obj.save(str(save_path))
        webbrowser.open(f'file://{save_path.absolute()}')
        return save_path
    
    def get_map_html(self, map_obj: folium.Map) -> str:
        """Get map as HTML string for embedding"""
        return map_obj._repr_html_()


if __name__ == "__main__":
    # Test map visualizer
    print("Testing Map Visualizer\n")
    
    # Create sample coordinates (example route in Bangalore)
    sample_coords = [
        (12.9352, 77.6245),  # Koramangala
        (12.9400, 77.6300),
        (12.9450, 77.6350),
        (12.9500, 77.6200),
        (12.9550, 77.6150),  # Indiranagar
    ]
    
    visualizer = RouteMapVisualizer()
    
    # Create single route map
    route_map = visualizer.create_route_map(
        coordinates=sample_coords,
        route_name="Morning Run",
        distance_km=5.2,
        elevation_gain=45
    )
    
    # Save and open
    save_path = visualizer.open_in_browser(route_map, "test_route.html")
    print(f"\nMap opened in browser: {save_path}")
