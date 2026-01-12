# Forest Analyzer

Forest Analyzer is a web-based geospatial analysis tool that estimates forest cover and deforestation across user-defined regions using satellite imagery and NDVI-based analysis.

Users can draw a bounding box anywhere in the world, select two time periods, and visualize vegetation change through interactive map overlays.

This project was built during **DeltaHacks**, McMaster University’s annual hackathon.

## Features
- Interactive map-based region selection
- NDVI-based forest estimation using Sentinel-2 imagery
- Year-over-year vegetation comparison
- Visualization of vegetation loss and gain
- Lightweight Flask backend with server-side geospatial computation

## Tech Stack
- **Languages:** Python, JavaScript, HTML, CSS  
- **Backend:** Flask  
- **Geospatial:** Google Earth Engine  
- **Mapping:** Leaflet.js  

## Notes
This tool estimates forest cover using NDVI as a proxy. Results may vary due to seasonality, cloud cover, land cover type, and imagery availability.

## Running Locally
```bash
python3 -m pip install -r requirements.txt
python3 app.py
