import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
from shapely.geometry import Point
from shapely.ops import unary_union
import cartopy.feature as cfeature
from Polar_Diagram import df as polar_df
from cost_config import ANGLE_LIMIT, OMEGA, GAMMA, ALPHA, BETA
from shapely.geometry import Point, LineString
import cartopy.feature as cfeature
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString


# --- Corrected Land Mask Loading and Handling ---
# Load land polygons once at startup (with buffer for safety)
LAND_POLYGONS = gpd.read_file(
    "/Noray/data/land-polygons-split-4326/land_polygons.shp",
    bbox=(-2.15, 36.8, -0.85, 37.95)
).buffer(0.00015)  # ~15m buffer

def load_land_mask():
    """Load and properly orient the land mask"""
    data = np.load("murcia_ultra_land_mask.npz")
    mask = data["mask"]
    lon_grid = data["lon_grid"]
    lat_grid = data["lat_grid"]
    
    # Ensure proper orientation (latitude increasing)
    if lat_grid[0] > lat_grid[-1]:
        mask = np.flipud(mask)
        lat_grid = np.flip(lat_grid)
    
    return mask, lon_grid, lat_grid

# Load the mask once at startup
LAND_MASK, LON_GRID, LAT_GRID = load_land_mask()

def verify_crossing(start, end):
    """Simple and efficient land crossing check using Shapely"""
    start_lonlat = (start[1], start[0])
    end_lonlat = (end[1], end[0]) 
    path = LineString([start_lonlat, end_lonlat])
    crosses = LAND_POLYGONS[LAND_POLYGONS.intersects(path)]
    
    if not crosses.empty:
        return True
    
    return False


def get_cardinal_direction(angle):
    angle = angle % 360
    if angle >= 337.5 or angle < 22.5:
        return 'N'
    elif 22.5 <= angle < 67.5:
        return 'NE'
    elif 67.5 <= angle < 112.5:
        return 'E'
    elif 112.5 <= angle < 157.5:
        return 'SE'
    elif 157.5 <= angle < 202.5:
        return 'S'
    elif 202.5 <= angle < 247.5:
        return 'SW'
    elif 247.5 <= angle < 292.5:
        return 'W'
    elif 292.5 <= angle < 337.5:
        return 'NW'

def extract_wind_info(ds, time, lat, lon):
    lat_idx = np.abs(ds['latitude'].values - lat).argmin()
    lon_idx = np.abs(ds['longitude'].values - lon).argmin()
    time_idx = np.abs(ds['valid_time'].values - np.datetime64(time)).argmin()

    U = ds['u10'][time_idx, lat_idx, lon_idx].values
    V = ds['v10'][time_idx, lat_idx, lon_idx].values

    wind_speed = np.sqrt(U**2 + V**2) * 1.94384
    # Since we are using mathematics, we need the formula that gives 0º as East, but in the get_cardinal_direction 
    # we use the formula 270-wind_dir to get the actual wind direction

    wind_dir = (np.rad2deg(np.arctan2(V, U)) + 360) % 360
    return wind_speed, wind_dir, get_cardinal_direction((270 - wind_dir) % 360)

def haversine(lat1, lon1, lat2, lon2):
    R = 3440.065
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def get_boat_speeds(wind_speed, df):
    wind_speeds = np.array([float(c) for c in df.columns])
    if wind_speed <= wind_speeds[0]:
        speeds = df.iloc[:, 0]
    elif wind_speed >= wind_speeds[-1]:
        speeds = df.iloc[:, -1]
    else:
        idx = np.searchsorted(wind_speeds, wind_speed) - 1
        w1, w2 = wind_speeds[idx], wind_speeds[idx + 1]
        weight = (wind_speed - w1) / (w2 - w1)
        v1, v2 = df.iloc[:, idx], df.iloc[:, idx + 1]
        speeds = v1 + weight * (v2 - v1)
    return pd.DataFrame({'Angles': df.index, 'Speed': speeds.values})

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia directa entre dos puntos
    """
    R = 3440.065  # Earth radius in nautical miles
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c  


def calculate_isochrone(current_positions, target_position, ds, df, date_time):
    current_positions = np.array(current_positions)
    polar_angles = np.array(df.index)
    
    # Pre-allocate arrays for better performance
    n_positions = len(current_positions)
    n_angles = len(polar_angles)
    total_points = n_positions * n_angles
    
    next_positions = np.zeros((total_points, 2))
    next_parents = np.zeros(total_points, dtype=int)
    boat_speeds = np.zeros(total_points)
    true_wind_angles = np.zeros(total_points)
    local_time = date_time

    distance_to_target = haversine_distance(current_positions[0][0], current_positions[0][1], target_position[0], target_position[1])

    # Adjust time step dynamically based on distance (in nautical miles)
    if distance_to_target < 0.5:
        time_step = 0.1  # Don't go lower than this
    elif distance_to_target < 2:
        time_step = 0.1
    else:
        time_step = 0.1# Default time step        
    if isinstance(current_positions, tuple):
        current_positions = [current_positions]

    idx = 0
    for pos_idx, (boat_x, boat_y) in enumerate(current_positions):
        # Get wind information at boat position
        local_wind_speed, local_wind_dir, w = extract_wind_info(ds, local_time, boat_x, boat_y)
        
        # Get boat speeds for this wind speed
        speeds_df = get_boat_speeds(local_wind_speed, df)
        local_time = local_time + timedelta(minutes=time_step*60)
        # For each angle in the polar diagram
        for angle_idx, twa in enumerate(polar_angles):
            # Get boat speed for this true wind angle
            boat_speed = speeds_df[speeds_df['Angles'] == twa]['Speed'].values[0]
            
            # Calculate sailing direction: wind direction + true wind angle
            sailing_angle = (local_wind_dir + 180 + twa) % 360
            sailing_angle_rad = np.deg2rad(sailing_angle)
            
            # Calculate new position after time_step hours
            new_x = boat_x + (boat_speed * time_step * np.cos(sailing_angle_rad))/ (60 * np.cos(np.deg2rad(boat_y)))  # Longitude shift
            new_y = boat_y + boat_speed * time_step * np.sin(sailing_angle_rad) / 60 
            
            # Store results
            next_positions[idx] = [new_x, new_y]
            next_parents[idx] = pos_idx
            boat_speeds[idx] = boat_speed
            true_wind_angles[idx] = twa
            local_time = local_time 
            idx += 1
    
    return next_positions, next_parents, boat_speeds, true_wind_angles, local_time

def is_target_crossed(current_pos, previous_pos, target_pos):
        """
        Mira si el barco ha atravesado el destino dinal. Usamos una tolerancia de 0.2, ya que es la que usamos cuando llegamos al destino final en find optimal path a star
        """
        x1, y1 = current_pos
        x2, y2 = previous_pos
        xt, yt = target_pos

        # Handle vertical line to avoid division by zero
        if x2 == x1:
            if xt == x1 and min(y1, y2) <= yt <= max(y1, y2):
                return True
            return False
        
        # Calculate slope and intercept
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        if np.isclose(yt, m * xt + b, atol=0.2) and min(x1, x2) <= xt <= max(x1, x2) and min(y1, y2) <= yt <= max(y1, y2):
            return True
        return False 

import heapq

def find_optimal_path(start_pos, target_pos, ds, df, date_time, max_angle_change=45):
    if isinstance(date_time, str):
        local_time = datetime.strptime(date_time, "%Y-%m-%dT%H:%M:%S")
    else:
        local_time = date_time

    heap = []
    heapq.heappush(heap, (0, tuple(start_pos)))
    node_data = {
        tuple(start_pos): {
            'cost': 0,
            'position': start_pos,
            'g': 0,
            'path': [],
            'distance': 0,
            'speeds': 0,
            'angles': [],
            'time': local_time
        }
    }

    visited = set()
    final_state = None
    step_counter = 0

    while heap:
        _, current_pos = heapq.heappop(heap)

        if current_pos in visited:
            continue
        visited.add(current_pos)

        state = node_data[current_pos]
        cost, g, path, distance, speeds, angles, current_time = (
            state['cost'], state['g'], state['path'], state['distance'],
            state['speeds'], state['angles'], state['time']
        )

        previous_pos = path[-1] if path else current_pos

        # 🏁 Check if target is reached directly
        if haversine_distance(current_pos[0], current_pos[1], target_pos[0], target_pos[1]) < 0.2 or \
            is_target_crossed(current_pos, previous_pos, target_pos):
            print("✅ Target reached!")
            final_state = state
            break

        path = path + [current_pos]

        next_positions, parents, next_speeds, next_angles, next_time = calculate_isochrone(
            np.array([current_pos]), target_pos, ds, df, current_time)

        for i, next_pos in enumerate(next_positions):
            next_pos_tuple = tuple(next_pos)
            if next_pos_tuple in visited:
                continue
            
            # ⛔ Skip nodes that cross land
            if verify_crossing(current_pos, next_pos):
                continue
            

            estimated_speed = next_speeds[i]
            if estimated_speed == 0:
                continue  # Avoid division by zero

            h = haversine_distance(next_pos[0], next_pos[1], target_pos[0], target_pos[1]) / estimated_speed
            d = haversine_distance(current_pos[0], current_pos[1], next_pos[0], next_pos[1])
            time_taken = d / estimated_speed
            new_time = current_time + timedelta(minutes=int(round(time_taken * 60))) if math.isfinite(time_taken) else current_time
            new_g = g + d / estimated_speed
            new_d = distance + d


            new_angle = next_angles[i]
            angle_penalty = 0
            if angles:
                angle_diff = abs(new_angle - angles[-1])
                angle_diff = min(angle_diff, 360 - angle_diff)
                if angle_diff > ANGLE_LIMIT:
                    angle_penalty = GAMMA * (angle_diff - ANGLE_LIMIT)

            f = OMEGA * h + angle_penalty + BETA/estimated_speed

            # Add to heap only if better
            if next_pos_tuple not in node_data or f < node_data[next_pos_tuple]['cost']:
                node_data[next_pos_tuple] = {
                    'cost': f,
                    'position': next_pos_tuple,
                    'g': new_g,
                    'path': path,
                    'distance': new_d,
                    'speeds': estimated_speed,
                    'angles': angles + [new_angle],
                    'time': new_time
                }
                heapq.heappush(heap, (f, next_pos_tuple))

    # 🚢 Final route
    if final_state:
        path = np.array(final_state['path'] + [target_pos])
        total_distance = final_state['distance']
        avg_speed = final_state['speeds'] if isinstance(final_state['speeds'], float) else np.mean(final_state['speeds'])
        total_time = final_state['time']
        angles = final_state['angles'] + [0]
        times = []

        for i in range(len(path)-1, -1, -1):
            times.insert(0, current_time)
            if i > 0:
                d = haversine_distance(path[i][0], path[i][1], path[i-1][0], path[i-1][1])
                s = avg_speed
                current_time -= timedelta(hours=d / s)

        df_optimal = pd.DataFrame({
            'Latitude': [p[0] for p in path],
            'Longitude': [p[1] for p in path],
            'Timestamp': times,
            'Angles': angles
        })

        df_optimal.to_csv("optimal_path.csv", index=False)
        print("✅ Optimal path saved to 'optimal_path.csv'")
        return df_optimal

    else:
        print("❌ No path found!")
        return pd.DataFrame()
    
