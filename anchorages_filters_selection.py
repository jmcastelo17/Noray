"""
Created on Mon Mar 10 12:16:01 2025

@author: castelo
"""
import math
import xarray as xr
import numpy as np
from anchorages_data import anchorages


def get_cardinal_direction(angle):
    # Normalize the angle to a value between 0 and 360
    angle = angle % 360
    
    # Cardinal directions based on angle ranges
    if angle >= 337.5 or angle < 22.5:
        return 'N'   # North
    elif 22.5 <= angle < 67.5:
        return 'NE'  # North-East
    elif 67.5 <= angle < 112.5:
        return 'E'   # East
    elif 112.5 <= angle < 157.5:
        return 'SE'  # South-East
    elif 157.5 <= angle < 202.5:
        return 'S'   # South
    elif 202.5 <= angle < 247.5:
        return 'SW'  # South-West
    elif 247.5 <= angle < 292.5:
        return 'W'   # West
    elif 292.5 <= angle < 337.5:
        return 'NW'  # North-West
    

#Calculation of distance between current location and other spots
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on Earth.
    Returns distance in nautical miles.
    """
    R = 3440  # Earth's radius in nautical miles
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c  # Distance in nautical miles

def extract_wind_info(ds, time, boat_x, boat_y):
    """
    Extracts wind speed and direction at the boat's position and computes the True Wind Angle (TWA).

    Parameters:
    X, Y (ndarray): Wind field grid coordinates.
    U_field, V_field (ndarray): Wind velocity components.
    boat_x, boat_y (float): Boat's current position.
    boat_heading (float): Boat's heading in degrees (0° = East, 90° = North).

    Returns:
    wind_speed (float): Wind speed at the boat's position.
    wind_direction (float): Wind direction in degrees (0° = East, 90° = North).
    TWA (float): True Wind Angle relative to the boat.
    """
    # Find the indices of the closest grid points
    lat_idx = np.abs(ds['latitude'].values - boat_x).argmin()
    lon_idx = np.abs(ds['longitude'].values - boat_y).argmin()
    time_idx = np.abs(ds['valid_time'].values - np.datetime64(time)).argmin()

    # Extract u10 and v10 at the specific time, latitude, and longitude
    U = ds['u10'][time_idx, lat_idx, lon_idx].values
    V = ds['v10'][time_idx, lat_idx, lon_idx].values

    # Compute wind speed and direction
    wind_speed = np.sqrt(U**2 + V**2) * 1.94384
    wind_direction = (np.rad2deg(np.arctan2(V, U)) + 360) % 360  # Convert to degrees (0-360)
    meteo_wind_direction = (270 - wind_direction) % 360
    cardinal_direction = (get_cardinal_direction(meteo_wind_direction))


    return wind_speed, wind_direction, cardinal_direction

def filter_anchorages(current_lat, current_lon, max_distance_nm, anchorages, min_rating, seabed_type, dinghy_needs, ds, date_time):
    filtered_anchorages = {}

    # Normalize input
    seabed_type = [s.lower() for s in seabed_type]
    dinghy_needs = [d.lower() for d in dinghy_needs]

    for name, details in anchorages.items():
        distance = haversine(current_lat, current_lon, details['latitude'], details['longitude'])

        if distance > max_distance_nm:
            print(f"⛔ {name}: too far ({distance:.2f} NM)")
            continue
        if details['rating'] < min_rating:
            print(f"⛔ {name}: rating too low ({details['rating']})")
            continue

        # Seabed check (skip if not matching and not "don't mind")
        if "don't mind" not in seabed_type:
            if not any(sb in details['seabed type'] for sb in seabed_type):
                print(f"⛔ {name}: seabed mismatch (user: {seabed_type}, has: {details['seabed type']})")
                continue

        # Dinghy check
        if "don't mind" not in dinghy_needs:
            if not any(dn in details['dinghy'] for dn in dinghy_needs):
                print(f"⛔ {name}: dinghy mismatch (user: {dinghy_needs}, has: {details['dinghy']})")
                continue

        # Wind and protection match
        wind_speed, wind_dir, cardinal = extract_wind_info(ds, date_time, details['latitude'], details['longitude'])
        if cardinal.lower() not in [d.lower() for d in details['protection']]:
            print(f"⛔ {name}: not protected from wind ({cardinal}, protects: {details['protection']})")
            continue

        print(f"✅ {name}: matched!")
        filtered_anchorages[name] = details
        filtered_anchorages[name]['distance_nm'] = round(distance, 2)

    return filtered_anchorages



def popping_most_like_spot(ds, date_time,location, max_distance_nm, seabed_type, dinghy_needs, rating):
    if location.lower() == "yes":
        current_location  = {
        "Club Náutico de Águilas": {
            "latitude": 37.4082,  
            "longitude": -1.5725
            }
        }
        current_lat = current_location["Club Náutico de Águilas"]["latitude"]
        current_lon = current_location["Club Náutico de Águilas"]["longitude"]
        # Get nearby anchorages
        filtered_anchorages = filter_anchorages(current_lat, current_lon, max_distance_nm, anchorages, rating, seabed_type, dinghy_needs, ds, date_time)
    
        # Display the filtered anchorages
        if filtered_anchorages:
            print("\nHere are the anchorages that match your preferences:")
            for name, details in filtered_anchorages.items():
                print(f"{name}: {details['distance_nm']} nm away, Rating: {details['rating']}")
        else:
            print("\nNo anchorages match your criteria.")
    else:
        print("\nIt seems you're not at Club Náutico de Águilas. Please enter a valid location.")
        
    destination = input("So, where should I take you?")
    return destination
