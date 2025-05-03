# Noray

Noray is a weather routing application built for recreational sailors. It suggests safe anchorages and optimal sailing routes based on wind conditions, user preferences, and boat performance data.

## Features
- Route optimization using wind data and polar diagrams
- Anchorage recommendation system
- Wind direction and protection analysis
- Streamlit-based interactive interface
- Folium-based nautical chart with OpenSeaMap overlays

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/jmcastelo17/Noray.git
cd Noray
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download land polygon shapefiles
Run this script to download the land polygons used to verify land crossings:
```bash
python download_landpolygons.py
```

### 4. Run the app
```bash
streamlit run app.py
```

## Notes
- The project currently uses static `.nc` wind data (included).
- Shapefiles are fetched on first run via `download_landpolygons.py`.
- Real-time integration and user personalization are part of future development.

## License
MIT (or specify your own)

---

Make sure to edit your GitHub repo description and add a screenshot or demo if you'd like to enhance presentation.

Let me know if you'd like the exact contents written into new files now!
