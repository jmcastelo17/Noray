# app.py
import streamlit as st
import xarray as xr
import datetime
from anchorages_data import anchorages
from anchorages_filters_selection import filter_anchorages
import pandas as pd
from streamlit_algorithm import find_optimal_path
from Polar_Diagram import df, data, angles, wind_speeds
import pydeck as pdk
import numpy as np
import matplotlib.pyplot as plt
from math import atan2, degrees
import folium
from streamlit_folium import st_folium
import branca

@st.cache_resource
def load_dataset():
    return xr.open_dataset(
        "data_stream-enda_stepType-instant.nc",
        engine="netcdf4"
    )

ds = load_dataset()

def get_cardinal_direction(degrees):
    """Convert degrees to cardinal direction"""
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    index = round(degrees / (360. / len(directions))) % len(directions)
    return directions[index]

def extract_wind_info(ds, time, lat, lon):
    """
    Extract wind speed (in knots), direction in degrees, and cardinal direction
    """
    lat_idx = np.abs(ds['latitude'].values - lat).argmin()
    lon_idx = np.abs(ds['longitude'].values - lon).argmin()
    time_idx = np.abs(ds['valid_time'].values - np.datetime64(time)).argmin()

    U = ds['u10'][time_idx, lat_idx, lon_idx].values
    V = ds['v10'][time_idx, lat_idx, lon_idx].values

    wind_speed = np.sqrt(U**2 + V**2) * 1.94384  # Convert to knots
    wind_direction = (np.rad2deg(np.arctan2(V, U)) + 360) % 360  
    meteo_wind_direction = (270 - wind_direction) % 360
    cardinal_direction = get_cardinal_direction(meteo_wind_direction)

    return wind_speed, wind_direction, meteo_wind_direction,  cardinal_direction

def plot_polar_diagram():
    """Function to plot the polar diagram using your existing code"""
    # Convert angles to radians
    angles_rad = np.radians(angles)

    # Create a polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')  # Set North (0 degrees) at the top
    ax.set_theta_direction(-1)  # Clockwise direction

    # Define colors for each wind speed (same as your code)
    colors = {
        6: 'blue',
        8: 'green',
        10: 'red',
        12: 'purple',
        14: 'orange',
        16: 'cyan',
        20: 'magenta',
        24: 'black'
    }

    # Plot each wind speed curve separately
    for i, wind_speed in enumerate(wind_speeds):
        # Get boat speeds for this wind speed
        boat_speeds = [row[i] for row in data]
        
        # Split the data into two parts: before 150° and after 210°
        mask_before = np.array(angles) <= 150
        mask_after = np.array(angles) >= 210
        
        # Plot first part (up to 150°)
        ax.plot(angles_rad[mask_before], np.array(boat_speeds)[mask_before], 
                color=colors[wind_speed], label=f'{wind_speed} kt Wind' if mask_before[0] else "")
        
        # Plot second part (from 210° onwards)
        ax.plot(angles_rad[mask_after], np.array(boat_speeds)[mask_after], 
                color=colors[wind_speed])

    # Customize the plot
    ax.set_title("Polar Diagram - Boat Performance")
    ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
    ax.grid(True)

    # Set the maximum radius based on the maximum boat speed
    max_speed = max(max(row) for row in data)
    ax.set_rmax(max_speed + 1)

    # Add radial labels
    ax.set_rticks(np.arange(2, int(max_speed) + 2, 2))
    ax.set_rlabel_position(90)

    plt.tight_layout()
    return fig

# --- Header ---
st.markdown("### NORAY ⚓")
st.markdown("## Welcome back, **Alfredo**")
st.markdown("#### Where are we heading today?")

# --- Location Selection ---
location = st.selectbox("From where?", ["Select an anchorage..."] + list(anchorages.keys()))

# --- Sailing Time ---
user_datetime = st.sidebar.time_input("Choose your sailing time", value=datetime.time(12, 0))
date_time = datetime.datetime.combine(datetime.date(2025, 3, 9), user_datetime).isoformat()

# --- Wind Information ---
if location != "Select an anchorage...":
    try:
        wind_speed, wind_dir_deg, meteo_wind_direction, wind_dir_card = extract_wind_info(
            ds,
            date_time,
            anchorages[location]['latitude'],
            anchorages[location]['longitude']
        )
        
        st.sidebar.markdown(f"**Current Wind at {location}**")
        st.sidebar.metric("Wind Speed", f"{wind_speed:.1f} knots")
        st.sidebar.metric("Direction", f"{wind_dir_card} ({meteo_wind_direction:.0f}°)")
        
        # Optional: Show wind arrow visualization
        st.sidebar.markdown(f"<div style='text-align:center; font-size:2em; transform: rotate({wind_dir_deg}deg)'>↑</div>", 
                           unsafe_allow_html=True)
        
    except Exception as e:
        st.sidebar.error(f"Error loading wind data: {str(e)}")
        # Debug information
        st.sidebar.write("Dataset dimensions:", list(ds.dims))
        st.sidebar.write("Dataset coordinates:", list(ds.coords))
        st.sidebar.write("First 5 time values:", ds['valid_time'].values[:5])

# --- Direct Destination Selection ---
# Handle recommendation confirmation BEFORE selectbox renders
if st.session_state.get("confirm_destination"):
    st.session_state.destination_select = st.session_state.temp_selected_destination
    del st.session_state.confirm_destination
    del st.session_state.temp_selected_destination

destination = st.selectbox(
    "Where to?",
    ["Select an anchorage..."] + list(anchorages.keys()),
    index=0,
    key="destination_select"
)

# If a destination has been confirmed through the recommendation flow
if "confirmed_destination_name" in st.session_state and "confirmed_destination_details" in st.session_state:
    confirmed_name = st.session_state.confirmed_destination_name
    confirmed_details = st.session_state.confirmed_destination_details

    st.success(f"You're headed to **{confirmed_name}**!")
    st.write(f"**Distance**: {confirmed_details['distance_nm']} NM")
    st.write(f"**Rating**: {confirmed_details['rating']} ⭐")
    st.write(f"**Seabed**: {', '.join(confirmed_details['seabed type'])}")
    st.write(f"**Protection**: {', '.join(confirmed_details['protection'])}")
    st.write(f"**Dinghy Access**: {', '.join(confirmed_details['dinghy'])}")

    # This ensures the routing works off the confirmed recommendation
    st.session_state.route_data = {
        "start_pos": (anchorages[location]['latitude'], anchorages[location]['longitude']),
        "destination": (confirmed_details['latitude'], confirmed_details['longitude']),
        "date_time": date_time
    }

# Normal dropdown-based flow (fallback)
elif destination != "Select an anchorage..." and location != "Select an anchorage...":
    st.success(f"You're headed to **{destination}**!")
    
    st.session_state.route_data = {
        "start_pos": (anchorages[location]['latitude'], anchorages[location]['longitude']),
        "destination": (anchorages[destination]['latitude'], anchorages[destination]['longitude']),
        "date_time": date_time
    }

    # Extract current wind direction
    try:
        _, _, meteo_wind_direction, wind_card = extract_wind_info(
            ds,
            date_time,
            anchorages[destination]['latitude'],
            anchorages[destination]['longitude']
        )

        # Check if wind direction is listed in protections
        protections = anchorages[destination].get("protection", [])
        if wind_card not in protections:
            st.warning(
                f"⚠️ Heads up! The anchorage **{destination}** may not be protected from **{wind_card}** winds."
            )
    except Exception as e:
        st.error(f"Could not evaluate wind protection: {e}")

st.markdown("**Or**")


# --- 'Let's find out' (Recommendation) ---
if "find_out_clicked" not in st.session_state:
    st.session_state.find_out_clicked = False

if st.button("Let's find out!", type="primary"):
    st.session_state.find_out_clicked = True

# --- Navigation Buttons ---
st.markdown("---")
cols = st.columns(2)
with cols[0]:
    if st.button("🌬️ Wind Prediction", use_container_width=True):
        st.markdown(
        """
        <a href="https://www.windy.com/station/wmo-08432?waves,36.427,-1.587,8" target="_blank">
            <button style='width: 100%; padding: 0.75em; font-size: 1rem; background-color: #0E1117; color: white; border: none; border-radius: 5px;'>
                🌬️ Wind Prediction
            </button>
        </a>
        """,
        unsafe_allow_html=True
)
    if st.button("📈 Past Routes", use_container_width=True):
        st.info("🚧 This feature is under development.")

with cols[1]:
    if st.button("📊 Polar Diagram", use_container_width=True):
        st.session_state.show_polar = True

    if st.button("🗺️ Nautical Chart", use_container_width=True):
        st.session_state.show_nautical_chart = True

# Show polar diagram if button was clicked
if st.session_state.get("show_polar", False):
    st.pyplot(plot_polar_diagram())
    if st.button("Close Polar Diagram"):
        st.session_state.show_polar = False

# --- Nautical Chart view (only start location)
if st.session_state.get("show_nautical_chart", False):
    if location != "Select an anchorage...":
        lat = anchorages[location]['latitude']
        lon = anchorages[location]['longitude']
        
        chart_map = folium.Map(location=[lat, lon], zoom_start=11, tiles="OpenStreetMap")
        folium.TileLayer(
            tiles='https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png',
            attr='OpenSeaMap',
            name='OpenSeaMap',
            overlay=False,
            control=True
        ).add_to(chart_map)

        folium.Marker(
            location=[lat, lon],
            popup="Start Location",
            icon=folium.Icon(color='green', icon='anchor', prefix='fa')
        ).add_to(chart_map)

        st.markdown("### 🗺️ Nautical Chart")
        st_folium(chart_map, width=700, height=500, returned_objects=[])

        if st.button("Close Nautical Chart"):
            st.session_state.show_nautical_chart = False
    else:
        st.warning("Please select a departure anchorage first.")


# --- Survey Form ---
if st.session_state.find_out_clicked:
    st.markdown("### 🧭 Let's Find the Perfect Anchorage for You")

    dinghy_selection = st.multiselect("Activities", ["Beach", "Restaurant", "Dock", "Water", "Don't mind"], key="dinghy")
    distance = st.slider("Max Distance (NM)", 1, 30, 10, key="distance")
    seabed_selection = st.multiselect("Seabed Type", ["Rock", "Sand", "Seagrass", "Don't mind"], key="seabed")
    min_rating = st.slider("Min Rating", 1, 5, 3, key="rating")

    if st.button("Check Route"):
        st.session_state.survey_data = {
            "dinghy": dinghy_selection,
            "distance": distance,
            "seabed": seabed_selection,
            "rating": min_rating
        }

        if location == "Select an anchorage...":
            st.warning("Please select a departure point.")
        else:
            # Get matches
            results = filter_anchorages(
                anchorages[location]["latitude"],
                anchorages[location]["longitude"],
                distance,
                anchorages,
                min_rating,
                seabed_selection,
                dinghy_selection,
                ds,
                date_time
            )

            if results:
                st.session_state.recommendation_results = results
                st.session_state.selected_result = list(results.keys())[0]  # Default to first
            else:
                st.warning("⚠️ No anchorages matched your criteria.")
                st.session_state.recommendation_results = None

    # If we have results, show the selector
    if "recommendation_results" in st.session_state and st.session_state.recommendation_results:
        st.markdown("### 🏝️ Anchorages Matching Your Preferences:")

        selected = st.selectbox(
            "Pick one to sail to:",
            options=list(st.session_state.recommendation_results.keys()),
            key="selected_result"
        )

        details = st.session_state.recommendation_results[selected]

        # Show characteristics of the selected anchorage
        st.write(f"**Distance**: {details['distance_nm']} NM")
        st.write(f"**Rating**: {details['rating']} ⭐")
        st.write(f"**Seabed**: {', '.join(details['seabed type'])}")
        st.write(f"**Protection**: {', '.join(details['protection'])}")
        st.write(f"**Dinghy Access**: {', '.join(details['dinghy'])}")

        # Confirm button: trigger flag and rerun
        if st.button("✅ Confirm This Destination"):
            st.session_state.confirm_destination = True
            st.session_state.temp_selected_destination = selected
            st.rerun()


if "route_data" in st.session_state:
    if st.button("🧭 Show Route"):
        route = st.session_state.route_data
        st.write("📍 Start Position:", route["start_pos"])
        st.write("📍 Destination:", route["destination"])
        st.write("🕒 Time:", route["date_time"])

        # Compute the optimal route
        route_df = find_optimal_path(route["start_pos"], route["destination"], ds, df, date_time)
        
        if route_df is None or route_df.empty:
            st.error("⚠️ No route data was returned.")

        if route_df is not None:
            # Fix headers if the first row is being interpreted as data
            if 0 in route_df.columns:
                route_df.columns = route_df.iloc[0]  # First row becomes header
                route_df = route_df[1:]              # Remove header row from data
                route_df.reset_index(drop=True, inplace=True)

            # Rename for Streamlit map compatibility
            route_df = route_df.rename(columns={"Latitude": "latitude", "Longitude": "longitude"})

            # 🖼️ Preview data
            st.markdown("📍 **Route Data Preview:**")
            st.dataframe(route_df)

            segments = pd.DataFrame({
                "from_lat": route_df["latitude"][:-1].values,
                "from_lon": route_df["longitude"][:-1].values,
                "to_lat": route_df["latitude"][1:].values,
                "to_lon": route_df["longitude"][1:].values
            })

            # Create map with proper OpenSeaMap tiles
        m = folium.Map(
            location=[route_df['latitude'].mean(), route_df['longitude'].mean()],
            zoom_start=10,
            tiles="OpenStreetMap"  # We'll add our own tile layer
        )

        # Add OpenSeaMap tile layer with proper attribution
        folium.TileLayer(
            tiles='https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png',
            attr='OpenSeaMap',
            name='OpenSeaMap',
            overlay=False,
            control=True
        ).add_to(m)

        # Add route line
        folium.PolyLine(
            locations=route_df[['latitude', 'longitude']].values.tolist(),
            color='#131B65',
            weight=4,
            opacity=0.8,
            popup="Optimal Route"
        ).add_to(m)

        # Add markers with custom icons
        folium.Marker(
            location=[route_df['latitude'].iloc[0], route_df['longitude'].iloc[0]],
            popup="Start",
            icon=folium.Icon(color='green', icon='anchor', prefix='fa')
        ).add_to(m)

        folium.Marker(
            location=[route_df['latitude'].iloc[-1], route_df['longitude'].iloc[-1]],
            popup="Destination",
            icon=folium.Icon(color='blue', icon='flag', prefix='fa')
        ).add_to(m)

        # Add layer control and fit bounds
        m.fit_bounds([[route_df['latitude'].min(), route_df['longitude'].min()], 
                     [route_df['latitude'].max(), route_df['longitude'].max()]])

        # Display the map using st_folium
        st_folium(m, width=700, height=500, returned_objects=[])

        # Download button
        st.download_button("📥 Download Route CSV", route_df.to_csv(index=False), "route.csv")

#streamlit run app.py


    
