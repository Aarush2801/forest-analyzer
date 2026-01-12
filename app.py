from flask import Flask, render_template, request
import ee

# Earth Engine init

ee.Initialize(project="studious-karma-482808-m2")

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  


CACHE = {}


def parse_bbox(bbox_str: str):
    parts = [p.strip() for p in bbox_str.split(",")]
    if len(parts) != 4 or any(p == "" for p in parts):
        raise ValueError("bbox must be 4 comma-separated numbers: min_lon,min_lat,max_lon,max_lat")
    min_lon, min_lat, max_lon, max_lat = map(float, parts)
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("bbox coordinates are invalid (min must be < max).")
    return min_lon, min_lat, max_lon, max_lat


def choose_scale(min_lon, min_lat, max_lon, max_lat):
    # Rough rectangle size in degrees^2 
    width = abs(max_lon - min_lon)
    height = abs(max_lat - min_lat)
    area_deg2 = width * height

    
    if area_deg2 > 3.0:
        return 240
    if area_deg2 > 1.0:
        return 120
    if area_deg2 > 0.25:
        return 60
    return 30


def get_ndvi(region, year: int, cloud_pct: int = 60, season: str = "jun-sep"):
    # Fewer months -> faster
    if season == "jun-sep":
        start = ee.Date.fromYMD(year, 6, 1)
        end = ee.Date.fromYMD(year, 9, 30)
    elif season == "all":
        start = ee.Date.fromYMD(year, 1, 1)
        end = ee.Date.fromYMD(year, 12, 31)
    else:
        start = ee.Date.fromYMD(year, 6, 1)
        end = ee.Date.fromYMD(year, 9, 30)

    img = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .select(["B4", "B8"])
        .median()
    )
    return img.normalizedDifference(["B8", "B4"]) 


def area_km2(mask_img, region, scale):
    """
    mask_img: ee.Image of 0/1 or boolean mask (masked pixels ignored)
    Returns area in km^2 by summing pixel area where mask is 1.
    """
    
    masked = ee.Image(1).updateMask(mask_img)
    area_img = ee.Image.pixelArea().updateMask(masked)  

    stats = area_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        bestEffort=True,
        maxPixels=1e9,
    )
    
    val = ee.Dictionary(stats).values().get(0)
    m2 = ee.Number(val)
    return m2.divide(1e6)  # km^2


def compute_phase1(bbox_str: str, year_before: int, year_after: int, season: str = "jun-sep"):
    min_lon, min_lat, max_lon, max_lat = parse_bbox(bbox_str)
    region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
    scale = choose_scale(min_lon, min_lat, max_lon, max_lat)

    # NDVI images
    ndvi_before = get_ndvi(region, year_before, cloud_pct=60, season=season)
    ndvi_after = get_ndvi(region, year_after, cloud_pct=60, season=season)

    # NDVI diff
    ndvi_diff = ndvi_after.subtract(ndvi_before)

    # Forest masks (NDVI proxy)
    threshold = 0.3
    forest_before = ndvi_before.gt(threshold)
    forest_after = ndvi_after.gt(threshold)
    lost_forest = forest_before.And(forest_after.Not())

    # Clip overlays to the bbox so they don't show as giant squares
    ndvi_before_c = ndvi_before.clip(region)
    ndvi_after_c = ndvi_after.clip(region)
    ndvi_diff_c = ndvi_diff.clip(region)
    lost_forest_c = lost_forest.clip(region)

    # ---- Phase 1 metrics (km²) ----
    forest_before_km2 = area_km2(forest_before, region, scale)
    forest_after_km2 = area_km2(forest_after, region, scale)
    loss_km2 = area_km2(lost_forest, region, scale)

    # Bring back to Python
    fb = float(forest_before_km2.getInfo() or 0.0)
    fa = float(forest_after_km2.getInfo() or 0.0)
    lk = float(loss_km2.getInfo() or 0.0)

    loss_percent = None
    if fb > 0:
        loss_percent = (lk / fb) * 100.0

    # ---- Tile URLs ----
    diff_vis = {"min": -0.5, "max": 0.5, "palette": ["red", "white", "green"]}
    ndvi_vis = {"min": 0.0, "max": 1.0, "palette": ["brown", "yellow", "green"]}
    lost_vis = {"min": 0, "max": 1, "palette": ["00000000", "ff0000"]}  # transparent -> red

    tile_urls = {
        "diff": ee.Image(ndvi_diff_c).getMapId(diff_vis)["tile_fetcher"].url_format,
        "before": ee.Image(ndvi_before_c).getMapId(ndvi_vis)["tile_fetcher"].url_format,
        "after": ee.Image(ndvi_after_c).getMapId(ndvi_vis)["tile_fetcher"].url_format,
        "lost": ee.Image(lost_forest_c).getMapId(lost_vis)["tile_fetcher"].url_format,
    }

    metrics = {
        "forest_before_km2": fb,
        "forest_after_km2": fa,
        "loss_km2": lk,
        "loss_percent": loss_percent,
        "used_scale": scale,
    }

    return metrics, tile_urls


@app.route("/", methods=["GET", "POST"])
def index():
    default_center = {"lat": -9.25, "lon": -59.25, "zoom": 10}
    default_bbox = "-59.5,-9.5,-59.0,-9.0"

    if request.method == "POST":
        bbox = request.form.get("bbox", "").strip()
        yb = int(request.form.get("year_before", "2018"))
        ya = int(request.form.get("year_after", "2023"))
        season = request.form.get("season", "jun-sep")

        if not bbox:
            return render_template(
                "index.html",
                center=default_center,
                bbox=default_bbox,
                year_before=yb,
                year_after=ya,
                season=season,
                metrics=None,
                tile_urls=None,
                error="Draw a rectangle first (bbox is empty).",
            )

        key = (bbox, yb, ya, season)
        try:
            if key in CACHE:
                metrics, tile_urls = CACHE[key]
            else:
                metrics, tile_urls = compute_phase1(bbox, yb, ya, season)
                CACHE[key] = (metrics, tile_urls)
        except Exception as e:
            return render_template(
                "index.html",
                center=default_center,
                bbox=bbox or default_bbox,
                year_before=yb,
                year_after=ya,
                season=season,
                metrics=None,
                tile_urls=None,
                error=f"Could not analyze that area: {e}",
            )

        
        error = None
        if metrics["forest_before_km2"] == 0 and metrics["forest_after_km2"] == 0:
            error = "No valid pixels found (clouds/no data). Try a smaller area, different season, or different years."

        return render_template(
            "index.html",
            center=default_center,
            bbox=bbox,
            year_before=yb,
            year_after=ya,
            season=season,
            metrics=metrics,
            tile_urls=tile_urls,
            error=error,
        )

    # GET
    return render_template(
        "index.html",
        center=default_center,
        bbox=default_bbox,
        year_before=2018,
        year_after=2023,
        season="jun-sep",
        metrics=None,
        tile_urls=None,
        error=None,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
