from pathlib import Path
import tomllib
import geopandas as gpd
import odc.geo.xr
import odc.stac
import rioxarray  # noqa: F401
from odc.geo.geom import Geometry
from pystac_client import Client
from shapely.geometry import mapping


# ============================================================
# CONFIGURATION
# ============================================================

with open('config.toml', 'rb') as f:
    config = tomllib.load(f)
   
date_range_2021 = config['data']['train_date_range_2021']
date_range_2023 = config['data']['train_date_range_2023']
date_range_2025 = config['data']['train_date_range_2025']

date_ranges = [date_range_2021, date_range_2023, date_range_2025]

training_aoi_path = config['data']['train_area_geo']
test_aoi_path = config['data']['test_area_geo']
aois = [training_aoi_path, test_aoi_path]
region_mapping_name = config['data']['region_mapping_name']
cloud_cover = config['options']['cloud_cover']

catalog_url = "https://earth-search.aws.element84.com/v1"

bands = ["coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "nir09", "swir16", "swir22",]

def download(aoi, date_range, alias):

    date_range = (date_range.split("/")[0], date_range.split("/")[1]) # - baseline flowering season image

    aoi_path = Path(f"{aoi}")
    
    output_dir = Path(f"data/tiles/{alias}/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (
    f"{region_mapping_name}_{date_range[0]}_{date_range[1]}_"
    f"{aoi_path.stem}.tif"
)

    # Sentinel-2 SCL classes to reject
    #
    # 3  = Cloud shadow
    # 8  = Cloud medium probability
    # 9  = Cloud high probability
    # 10 = Thin cirrus
    #
    # We keep:
    # 4 = Vegetation
    # 5 = Bare soil
    # 6 = Water
    # 7 = Unclassified
    #
    # 0, 1, 2 can also be considered invalid depending on
    # the application. We explicitly mask them below as well.

    BAD_SCL = [0, 1, 2, 3, 8, 9, 10, 11]

    # ============================================================
    # CONNECT TO STAC
    # ============================================================

    catalog = Client.open(catalog_url)

    print("=" * 70)
    print("LOADING AOI")
    print("=" * 70)

    gdf = gpd.read_file(aoi_path)

    if gdf.empty:
        raise ValueError(f"AOI is empty: {aoi_path}")

    if gdf.geometry.isna().any():
        raise ValueError("AOI contains null geometries.")

    if not gdf.geometry.is_valid.all():
        print("WARNING: AOI contains invalid geometries.")
        print("Attempting geometry repair...")

        gdf["geometry"] = gdf.geometry.make_valid()

    if gdf.crs is None:
        raise ValueError("AOI has no CRS.")

    print(f"AOI: {aoi_path}")
    print(f"Features: {len(gdf)}")
    print(f"CRS: {gdf.crs}")
    print(f"Bounds: {gdf.total_bounds}")

    centroid = gdf.geometry.union_all().centroid
    lon, lat = centroid.x, centroid.y
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >=0 else 32700 + zone
    CRS = f'EPSG:{str(epsg)}'

    # The GeoJSON supplied is CRS84 / lon-lat.
    # Normalize explicitly to EPSG:4326 for STAC.
    gdf = gdf.to_crs("EPSG:4326")

    # Merge into one geometry in case the AOI contains multiple features.
    aoi_geometry = gdf.geometry.union_all()

    geometry = Geometry(mapping(aoi_geometry), "EPSG:4326")


    # ============================================================
    # SEARCH SENTINEL-2
    # ============================================================

    print()
    print("=" * 70)
    print("SEARCHING SENTINEL-2")
    print("=" * 70)

    datetime_range = (f"{date_range[0]}/{date_range[1]}")

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=geometry,
        datetime=datetime_range,
        query={
            "eo:cloud_cover": {
                "lt": cloud_cover
            }
        }
    )

    items = list(search.items())

    print(
        f"Date range: {date_range[0]} -> {date_range[1]}"
    )

    print(f"Found {len(items)} Sentinel-2 items")

    if not items:
        raise RuntimeError("No Sentinel-2 items found for the AOI/date range.")


    # Sort chronologically
    items = sorted(items, key=lambda item: item.datetime)

    for item in items:
        cloud = item.properties.get("eo:cloud_cover", "unknown")

        print(
            f"{item.id} | "
            f"{item.datetime} | "
            f"cloud={cloud}"
        )


    # ============================================================
    # LOAD SENTINEL-2
    # ============================================================

    print()
    print("=" * 70)
    print("LOADING IMAGERY")
    print("=" * 70)

    # IMPORTANT:
    # scl is loaded in addition to the spectral bands because
    # it is required for cloud/shadow masking.

    load_bands = bands + ["scl"]

    data = odc.stac.load(
        items,
        bands=load_bands,
        geopolygon=geometry,
        resolution=10,
        chunks={},
        groupby="solar_day",
    )

    print(data)

    print()
    print("Dates:")
    print(data.time.values)

    print()
    print("CRS:")
    print(data.odc.crs)


    # ============================================================
    # APPLY SCL CLOUD / SHADOW MASK
    # ============================================================

    print()
    print("=" * 70)
    print("APPLYING SCL CLOUD / SHADOW MASK")
    print("=" * 70)

    # SCL is categorical, so we don't want to include it
    # in the median calculation.

    valid_pixels = ~data.scl.isin(BAD_SCL)

    masked = data[bands]
    # masked = data[bands].where(valid_pixels)

    print("Masked spectral bands:")
    print(masked)


    # ============================================================
    # CREATE MEDIAN COMPOSITE
    # ============================================================

    print()
    print("=" * 70)
    print("CREATING MEDIAN COMPOSITE")
    print("=" * 70)

    result = masked.median(
        dim="time",
        skipna=True
    )

    print("Composite:")
    print(result)


    # ============================================================
    # CONVERT TO MULTIBAND DATAARRAY
    # ============================================================

    # Convert:
    #
    # blue       (y, x)
    # green      (y, x)
    # red        (y, x)
    # ...
    #
    # into:
    #
    # band       (y, x)

    result = result[bands].to_array(
        dim="band"
    )

    print()
    print("Multiband result:")
    print(result)

    print()
    print("Dimensions:")
    print(result.dims)


    # ============================================================
    # RESTORE CRS FOR RIOXARRAY
    # ============================================================

    print()
    print("=" * 70)
    print("RESTORING CRS")
    print("=" * 70)

    # odc.stac knows the CRS, but after xarray operations
    # the rioxarray accessor may no longer have the CRS attached.



    result = result.rio.write_crs(
        f"{CRS}",
        inplace=False
    )

    print(
        f"Result CRS: {result.rio.crs}"
    )


    # ============================================================
    # CLIP TO ACTUAL AOI POLYGON
    # ============================================================

    print()
    print("=" * 70)
    print("CLIPPING TO STUDY AREA")
    print("=" * 70)

    # Reproject AOI from EPSG:4326 to the
    # Sentinel-2 working CRS.
    gdf_projected = gdf.to_crs(f"{CRS}")

    print(
        f"AOI projected CRS: {gdf_projected.crs}"
    )

    print(
        f"AOI projected bounds: "
        f"{gdf_projected.total_bounds}"
    )

    # Clip to the actual polygon.
    #
    # This is NOT a bounding-box clip.
    # drop=True removes pixels outside the polygon.

    result = result.rio.clip(
        gdf_projected.geometry,
        gdf_projected.crs,
        drop=True
    )

    print()
    print("Clipped result:")
    print(result)

    print()
    print(
        f"Clipped dimensions: {result.dims}"
    )


    # ============================================================
    # SAVE AS CLOUD OPTIMIZED GEOTIFF
    # ============================================================

    print()
    print("=" * 70)
    print("SAVING COG")
    print("=" * 70)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    odc.geo.xr.write_cog(
        result,
        fname=str(output_path),
        overwrite=True,
    )

    print()
    print(f"Saved:")
    print(output_path)

    print()
    print("=" * 70)
    print("COMPOSITE COMPLETE")
    print("=" * 70)
    
if __name__ == '__main__':
    for i in aois:
        alias = 'test' if 'test' in i else 'training'
        for j in date_ranges:
            download(i, j, alias)