import os
import shelve
import json
import io
import csv
from flask import Flask, render_template, request, Response
import ee

GEE_PROJECT = os.environ.get("GEE_PROJECT", "studious-karma-482808-m2")
ee.Initialize(project=GEE_PROJECT)

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

CACHE_FILE = "gee_cache"
YEAR_MIN, YEAR_MAX = 2000, 2024
TIME_SERIES_YEARS = list(range(2017, 2025))


def cache_get(key_tuple):
    with shelve.open(CACHE_FILE) as db:
        return db.get(json.dumps(list(key_tuple)))


def cache_set(key_tuple, value):
    with shelve.open(CACHE_FILE) as db:
        db[json.dumps(list(key_tuple))] = value


def parse_bbox(bbox_str: str):
    parts = [p.strip() for p in bbox_str.split(",")]
    if len(parts) != 4 or any(p == "" for p in parts):
        raise ValueError("bbox must be 4 comma-separated numbers: min_lon,min_lat,max_lon,max_lat")
    min_lon, min_lat, max_lon, max_lat = map(float, parts)
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("bbox coordinates are invalid (min must be < max).")
    return min_lon, min_lat, max_lon, max_lat


def choose_scale(min_lon, min_lat, max_lon, max_lat):
    area_deg2 = abs(max_lon - min_lon) * abs(max_lat - min_lat)
    if area_deg2 > 3.0:
        return 240
    if area_deg2 > 1.0:
        return 120
    if area_deg2 > 0.25:
        return 60
    return 30


def get_ndvi(region, year: int, cloud_pct: int = 60, season: str = "jun-sep"):
    if season == "all":
        start, end = ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year, 12, 31)
    else:
        start, end = ee.Date.fromYMD(year, 6, 1), ee.Date.fromYMD(year, 9, 30)
    img = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .select(["B4", "B8"])
        .median()
    )
    return img.normalizedDifference(["B8", "B4"])


def get_forest_mask_hansen(year: int):
    """Binary forest mask from Hansen GFC v1.11 (tree cover ≥30%, accounting for loss up to year)."""
    hansen = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
    forest_2000 = hansen.select("treecover2000").gt(30)
    lossyear = hansen.select("lossyear")
    years_since_2000 = year - 2000
    if years_since_2000 <= 0:
        return forest_2000
    lost_by_year = lossyear.gt(0).And(lossyear.lte(years_since_2000))
    return forest_2000.And(lost_by_year.Not())


def area_km2_batch(masks: dict, region, scale: int) -> dict:
    """Compute areas (km²) for multiple binary masks in one GEE round-trip."""
    features = []
    for name, mask in masks.items():
        area_img = ee.Image.pixelArea().updateMask(ee.Image(1).updateMask(mask))
        stats = area_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=scale,
            bestEffort=True,
            maxPixels=1e9,
        )
        features.append(ee.Feature(None, {"name": name, "area_m2": stats.values().get(0)}))
    info = ee.FeatureCollection(features).getInfo()
    return {
        f["properties"]["name"]: float(f["properties"].get("area_m2") or 0) / 1e6
        for f in info["features"]
    }


def compute_time_series(region, scale: int, season: str = "jun-sep") -> list:
    """Return mean NDVI per year for TIME_SERIES_YEARS in one GEE round-trip."""
    features = []
    for year in TIME_SERIES_YEARS:
        ndvi = get_ndvi(region, year, season=season)
        stats = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=scale,
            bestEffort=True,
            maxPixels=1e9,
        )
        features.append(ee.Feature(None, {"year": year, "ndvi": stats.get("nd")}))
    info = ee.FeatureCollection(features).getInfo()
    pts = [
        {"year": int(f["properties"]["year"]), "ndvi": round(float(f["properties"]["ndvi"]), 4)}
        for f in info["features"]
        if f["properties"].get("ndvi") is not None
    ]
    return sorted(pts, key=lambda x: x["year"])


def compute_phase1(bbox_str: str, year_before: int, year_after: int, season: str = "jun-sep"):
    min_lon, min_lat, max_lon, max_lat = parse_bbox(bbox_str)
    region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
    scale = choose_scale(min_lon, min_lat, max_lon, max_lat)

    # NDVI imagery (for visualization)
    ndvi_before = get_ndvi(region, year_before, season=season).clip(region)
    ndvi_after = get_ndvi(region, year_after, season=season).clip(region)
    ndvi_diff = ndvi_after.subtract(ndvi_before)

    # Forest masks — Hansen (accurate) for 2000–2023, NDVI proxy otherwise
    use_hansen = YEAR_MIN <= year_before <= 2023 and YEAR_MIN <= year_after <= 2023
    if use_hansen:
        forest_before = get_forest_mask_hansen(year_before).clip(region)
        forest_after = get_forest_mask_hansen(year_after).clip(region)
    else:
        forest_before = ndvi_before.gt(0.3)
        forest_after = ndvi_after.gt(0.3)

    lost_forest = forest_before.And(forest_after.Not())

    # All three areas in one GEE call
    areas = area_km2_batch(
        {"forest_before": forest_before, "forest_after": forest_after, "lost_forest": lost_forest},
        region, scale,
    )
    fb, fa, lk = areas["forest_before"], areas["forest_after"], areas["lost_forest"]

    tile_urls = {
        "diff": ee.Image(ndvi_diff).getMapId(
            {"min": -0.5, "max": 0.5, "palette": ["red", "white", "green"]}
        )["tile_fetcher"].url_format,
        "before": ee.Image(ndvi_before).getMapId(
            {"min": 0.0, "max": 1.0, "palette": ["brown", "yellow", "green"]}
        )["tile_fetcher"].url_format,
        "after": ee.Image(ndvi_after).getMapId(
            {"min": 0.0, "max": 1.0, "palette": ["brown", "yellow", "green"]}
        )["tile_fetcher"].url_format,
        "lost": ee.Image(lost_forest).getMapId(
            {"min": 0, "max": 1, "palette": ["00000000", "ff0000"]}
        )["tile_fetcher"].url_format,
    }

    metrics = {
        "forest_before_km2": fb,
        "forest_after_km2": fa,
        "loss_km2": lk,
        "loss_percent": (lk / fb * 100.0) if fb > 0 else None,
        "used_scale": scale,
        "used_hansen": use_hansen,
    }

    time_series = compute_time_series(region, scale, season)
    return metrics, tile_urls, time_series


@app.route("/", methods=["GET", "POST"])
def index():
    default_center = {"lat": -9.25, "lon": -59.25, "zoom": 10}
    default_bbox = "-59.5,-9.5,-59.0,-9.0"
    ctx = dict(center=default_center, metrics=None, tile_urls=None, time_series=None, error=None)

    if request.method == "POST":
        bbox = request.form.get("bbox", "").strip()
        yb = int(request.form.get("year_before", "2018"))
        ya = int(request.form.get("year_after", "2023"))
        season = request.form.get("season", "jun-sep")
        ctx.update(bbox=bbox or default_bbox, year_before=yb, year_after=ya, season=season)

        if not bbox:
            ctx["error"] = "Draw a rectangle first (bbox is empty)."
            return render_template("index.html", **ctx)

        if not (YEAR_MIN <= yb <= YEAR_MAX and YEAR_MIN <= ya <= YEAR_MAX):
            ctx["error"] = f"Years must be between {YEAR_MIN} and {YEAR_MAX}."
            return render_template("index.html", **ctx)

        if ya <= yb:
            ctx["error"] = "After year must be greater than before year."
            return render_template("index.html", **ctx)

        key = (bbox, yb, ya, season)
        try:
            cached = cache_get(key)
            if cached:
                metrics, tile_urls, time_series = cached
            else:
                metrics, tile_urls, time_series = compute_phase1(bbox, yb, ya, season)
                cache_set(key, (metrics, tile_urls, time_series))
        except Exception as e:
            ctx["error"] = f"Could not analyze that area: {e}"
            return render_template("index.html", **ctx)

        if metrics["forest_before_km2"] == 0 and metrics["forest_after_km2"] == 0:
            ctx["error"] = "No valid pixels found (clouds/no data). Try a smaller area, different season, or different years."

        ctx.update(bbox=bbox, metrics=metrics, tile_urls=tile_urls, time_series=time_series)
        return render_template("index.html", **ctx)

    ctx.update(bbox=default_bbox, year_before=2018, year_after=2023, season="jun-sep")
    return render_template("index.html", **ctx)


@app.route("/export.csv")
def export_csv():
    bbox = request.args.get("bbox", "")
    season = request.args.get("season", "jun-sep")
    try:
        yb = int(request.args.get("year_before", "2018"))
        ya = int(request.args.get("year_after", "2023"))
    except ValueError:
        return "Invalid year parameters.", 400

    cached = cache_get((bbox, yb, ya, season))
    if not cached:
        return "No cached results. Run the analysis first.", 404

    metrics, _, time_series = cached

    output = io.StringIO()
    w = csv.writer(output)
    w.writerow(["bbox", bbox])
    w.writerow(["year_before", yb])
    w.writerow(["year_after", ya])
    w.writerow(["season", season])
    w.writerow([])
    w.writerow(["metric", "value"])
    w.writerow(["forest_before_km2", f"{metrics['forest_before_km2']:.4f}"])
    w.writerow(["forest_after_km2", f"{metrics['forest_after_km2']:.4f}"])
    w.writerow(["loss_km2", f"{metrics['loss_km2']:.4f}"])
    lp = metrics["loss_percent"]
    w.writerow(["loss_percent", f"{lp:.4f}" if lp is not None else "N/A"])
    w.writerow(["scale_m", metrics["used_scale"]])
    w.writerow(["method", "Hansen GFC" if metrics.get("used_hansen") else "NDVI threshold"])
    w.writerow([])
    w.writerow(["year", "mean_ndvi"])
    for pt in time_series:
        w.writerow([pt["year"], pt["ndvi"]])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=forest_analysis_{yb}_{ya}.csv"},
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
