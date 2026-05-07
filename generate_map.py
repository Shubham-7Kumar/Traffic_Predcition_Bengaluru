import pandas as pd
import folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import pickle
import requests

def main():
    print("Loading data...")
    df = pd.read_csv('Banglore_traffic_Dataset.csv')
    
    # Unique areas and roads
    areas = df['Area Name'].unique().tolist()
    roads = df['Road/Intersection Name'].unique().tolist()
    
    # Initialize Geocoder
    geolocator = Nominatim(user_agent="bengaluru_traffic_app")
    
    coords = {}
    
    # Fetch coordinates function
    def get_coords(location_name):
        if location_name in coords:
            return coords[location_name]
        try:
            # specifically search in bangalore
            loc = geolocator.geocode(f"{location_name}, Bengaluru, India")
            if loc:
                coords[location_name] = (loc.latitude, loc.longitude)
                return (loc.latitude, loc.longitude)
        except:
            pass
        return None

    print("Fetching coordinates for Areas...")
    for area in areas:
        get_coords(area)
        time.sleep(1) # respectful delay

    print("Fetching coordinates for Roads/Intersections...")
    for road in roads:
        get_coords(road)
        time.sleep(1)

    # Some might fail, let's hardcode fallbacks for known bangalore locations if they fail
    fallbacks = {
        'Indiranagar': (12.9784, 77.6408),
        'Whitefield': (12.9698, 77.7499),
        'Koramangala': (12.9352, 77.6245),
        'M.G. Road': (12.9719, 77.6010),
        'Jayanagar': (12.9304, 77.5811),
        'Hebbal': (13.0354, 77.5988),
        'Yeshwanthpur': (13.0279, 77.5409),
        'Electronic City': (12.8452, 77.6602),
        '100 Feet Road': (12.9784, 77.6408),
        'CMH Road': (12.9790, 77.6384),
        'Marathahalli Bridge': (12.9569, 77.6983),
        'ITPL Main Road': (12.9840, 77.7378),
        'Sony World Junction': (12.9370, 77.6259),
        'Sarjapur Road': (12.9238, 77.6508),
        'Trinity Circle': (12.9730, 77.6166),
        'Anil Kumble Circle': (12.9772, 77.6006),
        'Jayanagar 4th Block': (12.9290, 77.5824),
        'South End Circle': (12.9382, 77.5800),
        'Hebbal Flyover': (13.0360, 77.5971),
        'Ballari Road': (13.0475, 77.5925),
        'Yeshwanthpur Circle': (13.0248, 77.5385),
        'Tumkur Road': (13.0425, 77.5255),
        'Silk Board Junction': (12.9176, 77.6238),
        'Hosur Road': (12.9067, 77.6322)
    }

    for loc in areas + roads:
        if loc not in coords or coords[loc] is None:
            coords[loc] = fallbacks.get(loc, (12.9716, 77.5946)) # Default to Bengaluru center

    print("Generating Map...")
    # Create Folium Map centered on Bengaluru
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=11)

    # Plot Areas as Markers
    for area in areas:
        lat, lon = coords[area]
        folium.Marker(
            location=[lat, lon],
            popup=f"Area: {area}",
            icon=folium.Icon(color='darkblue', icon='car', prefix='fa')
        ).add_to(m)

    # Calculate distances and draw red lines
    distances = {}
    
    unique_pairs = df[['Area Name', 'Road/Intersection Name']].drop_duplicates()
    for _, row in unique_pairs.iterrows():
        area = row['Area Name']
        road = row['Road/Intersection Name']
        
        area_coords = coords[area]
        road_coords = coords[road]
        
        # Calculate Geodesic distance in km
        dist = geodesic(area_coords, road_coords).kilometers
        distances[f"{area}_{road}"] = dist
        
        # Fetch actual route via OSRM
        route_locations = [area_coords, road_coords]
        try:
            # OSRM expects longitude,latitude
            osrm_url = f"http://router.project-osrm.org/route/v1/driving/{area_coords[1]},{area_coords[0]};{road_coords[1]},{road_coords[0]}?overview=full&geometries=geojson"
            response = requests.get(osrm_url)
            data = response.json()
            if data.get('code') == 'Ok':
                route_coords = data['routes'][0]['geometry']['coordinates']
                # Folium expects latitude,longitude
                route_locations = [[lat, lon] for lon, lat in route_coords]
            else:
                print(f"OSRM Routing failed for {area} to {road}: {data.get('code')}")
        except Exception as e:
            print(f"OSRM Request failed for {area} to {road}: {e}")
            
        time.sleep(0.5) # Rate limiting for public OSRM API
        
        # Draw line
        folium.PolyLine(
            locations=route_locations,
            color='#4facfe',
            weight=3,
            opacity=0.8,
            popup=f"{area} to {road}: {dist:.2f} km"
        ).add_to(m)
        
        # Mark road
        folium.CircleMarker(
            location=road_coords,
            radius=6,
            popup=road,
            color='#00f2fe',
            fill=True,
            fillColor='#00f2fe',
            fillOpacity=1
        ).add_to(m)

    # Save Map
    map_file = 'bangalore_traffic_map.html'
    m.save(map_file)
    print(f"Map saved to {map_file}")

    # Save distances
    with open('distances.pkl', 'wb') as f:
        pickle.dump(distances, f)
    print("Distances saved to distances.pkl")

if __name__ == "__main__":
    main()
