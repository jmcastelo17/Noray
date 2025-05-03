import os
import requests
import zipfile

# URL of the land polygons data from OpenStreetMap
URL = "https://osmdata.openstreetmap.de/download/land-polygons-split-4326.zip"
OUTPUT_DIR = "data"
ZIP_PATH = "data/land-polygons.zip"

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# Step 1: Download the ZIP file
print("Downloading land polygons...")
response = requests.get(URL, stream=True)
with open(ZIP_PATH, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
print("Download completed.")

# Step 2: Extract the ZIP file
print("Extracting files...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(OUTPUT_DIR)

print(f"Files extracted to: {OUTPUT_DIR}")

# Optional: Clean up zip file
os.remove(ZIP_PATH)
print("Cleanup done.")