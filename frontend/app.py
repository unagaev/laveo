
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import rasterio
import folium
import matplotlib.pyplot as plt

from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)
from rasterio.enums import Resampling as RasterioResampling
from rasterio.transform import array_bounds
from streamlit_folium import st_folium


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RASTER_PATH = ROOT / "data" / "tiles" / "predictions" / "images" / "Dobrich_2021-07-01_2021-07-03_Dobrich_Province_Boundaries.tif"

VECTOR_PATHS = {
    2021: ROOT / "outputs" / "postprocessing" / "2021" / "final_fields.gpkg",
    2023: ROOT / "outputs" / "postprocessing" / "2023" / "final_fields.gpkg",
    2025: ROOT / "outputs" / "postprocessing" / "2025" / "final_fields.gpkg",
}

APP_NAME = "LAVEO"
APP_SUBTITLE = "Lavender Area Vectorisation with Earth Observation"

# Map appearance
POLYGON_COLOR = "#9B59D0"
POLYGON_FILL = "#A66BE0"
POLYGON_FILL_OPACITY = 0.38

# Optional: set this to the layer name if a GeoPackage has multiple layers.
# Leave as None to use the default layer.
GPKG_LAYER = None


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="LAVEO | Lavender Mapping",
    page_icon="🪻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background-color: #F7F7FB;
        }

        [data-testid="stSidebar"] {
            background-color: #211B32;
        }

        [data-testid="stSidebar"] * {
            color: #F5F1FC;
        }

        .brand-title {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #7650A6;
            margin-bottom: 0;
        }

        .brand-subtitle {
            color: #777184;
            font-size: 0.95rem;
            margin-top: 0;
            margin-bottom: 1.3rem;
        }

        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #30283D;
            margin-bottom: 0.35rem;
        }

        .muted {
            color: #777184;
            font-size: 0.9rem;
        }

        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E9E3F1;
            padding: 15px 16px;
            border-radius: 12px;
        }

        div[data-testid="stMetricLabel"] {
            color: #70677D;
        }

        div[data-testid="stMetricValue"] {
            color: #30283D;
        }

        .stButton button {
            border-radius: 8px;
        }

        hr {
            border-color: #E8E1F0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_vectors(path_string: str, layer=None):
    """Load a GeoPackage and return a valid GeoDataFrame."""
    path = Path(path_string)

    if not path.exists():
        return None, f"File not found: {path}"

    try:
        if layer:
            gdf = gpd.read_file(path, layer=layer)
        else:
            gdf = gpd.read_file(path)

        if gdf.empty:
            return None, f"No features found in {path.name}."

        if gdf.crs is None:
            return None, f"The GeoPackage has no CRS defined: {path.name}"

        gdf = gdf[gdf.geometry.notna()].copy()
        gdf = gdf[~gdf.geometry.is_empty].copy()

        if gdf.empty:
            return None, f"No valid geometries found in {path.name}."

        # Repair invalid geometry where possible.
        invalid = ~gdf.geometry.is_valid
        if invalid.any():
            gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].make_valid()

        return gdf, None

    except Exception as exc:
        return None, f"Could not read {path.name}: {exc}"


@st.cache_data(show_spinner=False)
def raster_to_overlay(path_string: str, max_dimension: int = 1800):
    """
    Prepare a raster for a Folium ImageOverlay.

    Reprojects the raster into EPSG:4326, applies a contrast stretch,
    and makes NoData/invalid pixels transparent.

    Returns:
        RGBA image, geographic bounds [[south, west], [north, east]],
        and an error message if processing failed.
    """
    path = Path(path_string)

    if not path.exists():
        return None, None, f"Raster not found: {path}"

    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                return None, None, "The raster has no CRS defined."

            if src.count < 3:
                return (
                    None,
                    None,
                    f"Expected at least 3 raster bands; found {src.count}.",
                )

            # Downsample large rasters to keep the web map responsive.
            scale = min(
                max_dimension / src.width,
                max_dimension / src.height,
                1.0,
            )

            dst_width = max(1, int(src.width * scale))
            dst_height = max(1, int(src.height * scale))

            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,
                "EPSG:4326",
                dst_width,
                dst_height,
                *src.bounds,
                dst_width=dst_width,
                dst_height=dst_height,
            )

            # Read the first three bands and their validity masks.
            source_rgb = src.read(
                [4, 3, 2],
                out_shape=(3, dst_height, dst_width),
                resampling=RasterioResampling.bilinear,
                masked=True,
            )

            source_data = np.asarray(source_rgb.filled(np.nan), dtype=np.float32)
            source_mask = np.all(~np.ma.getmaskarray(source_rgb), axis=0)

            # Destination arrays in geographic coordinates.
            dst_rgb = np.full(
                (3, dst_height, dst_width),
                np.nan,
                dtype=np.float32,
            )

            for band_index in range(3):
                reproject(
                    source=source_data[band_index],
                    destination=dst_rgb[band_index],
                    src_transform=src.transform
                    * src.transform.scale(
                        src.width / source_data.shape[2],
                        src.height / source_data.shape[1],
                    ),
                    src_crs=src.crs,
                    src_nodata=np.nan,
                    dst_transform=dst_transform,
                    dst_crs="EPSG:4326",
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )

            # Reproject the source validity mask separately.
            # This preserves transparent NoData areas.
            dst_valid = np.zeros(
                (dst_height, dst_width),
                dtype=np.uint8,
            )

            reproject(
                source=source_mask.astype(np.uint8),
                destination=dst_valid,
                src_transform=src.transform
                * src.transform.scale(
                    src.width / source_data.shape[2],
                    src.height / source_data.shape[1],
                ),
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs="EPSG:4326",
                dst_nodata=0,
                resampling=Resampling.nearest,
            )

            # Build an RGBA image with transparent invalid pixels.
            rgba = np.zeros((dst_height, dst_width, 4), dtype=np.uint8)
            valid = (dst_valid > 0) & np.all(np.isfinite(dst_rgb), axis=0)

            if not valid.any():
                return None, None, (
                    "No valid pixels found in the raster. "
                    "Check its bands, NoData values and masks."
                )

            # Independent percentile stretch per band for display.
            for band_index in range(3):
                band = dst_rgb[band_index]
                band_valid = valid & np.isfinite(band)

                if not band_valid.any():
                    continue

                low, high = np.percentile(
                    band[band_valid],
                    [2, 98],
                )

                if high <= low:
                    high = low + 1.0

                scaled = np.clip((band - low) / (high - low), 0, 1)
                scaled[~np.isfinite(scaled)] = 0

                rgba[:, :, band_index] = (scaled * 255).astype(np.uint8)

            rgba[:, :, 3] = np.where(valid, 255, 0).astype(np.uint8)

            west, south, east, north = array_bounds(
                dst_height,
                dst_width,
                dst_transform,
            )

            bounds = [[south, west], [north, east]]

            return rgba, bounds, None

    except Exception as exc:
        return None, None, f"Could not process raster: {exc}"


# ============================================================
# AREA CALCULATIONS
# ============================================================

def calculate_area_hectares(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Calculate feature areas in hectares using a suitable projected CRS."""
    if gdf.empty:
        return pd.Series(dtype=float)

    try:
        # Estimate a local UTM CRS for area calculation.
        projected_crs = gdf.estimate_utm_crs()
        if projected_crs:
            projected = gdf.to_crs(projected_crs)
            return projected.geometry.area / 10_000
    except Exception:
        pass

    # Fallback for datasets already using a projected CRS.
    if gdf.crs and gdf.crs.is_projected:
        return gdf.geometry.area / 10_000

    return pd.Series(np.nan, index=gdf.index, dtype=float)


@st.cache_data(show_spinner=False)
def load_all_vectors():
    """Load all available years and calculate area values."""
    result = {}
    errors = {}

    for year, path in VECTOR_PATHS.items():
        gdf, error = load_vectors(str(path), GPKG_LAYER)

        if error:
            errors[year] = error
            continue

        gdf = gdf.copy()
        gdf["area_ha"] = calculate_area_hectares(gdf)
        result[year] = gdf

    return result, errors


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="font-size: 2rem; font-weight: 800; letter-spacing: .08em;">
            🪻 LAVEO
        </div>
        <div style="font-size: .85rem; color: #C8B9DE; margin-bottom: 1.5rem;">
            Lavender mapping from Earth observation
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["Map", "Statistics", "Comparison", "About"],
        label_visibility="collapsed",
    )

    st.divider()

    st.caption("STUDY AREA")
    st.markdown("**Dobrich Province, Bulgaria**")
    st.caption("Sentinel-2 · 10 m · ResUNet")


# ============================================================
# LOAD DATA
# ============================================================

vectors_by_year, vector_errors = load_all_vectors()

available_years = sorted(vectors_by_year.keys())

if not available_years:
    st.markdown(
        f'<div class="brand-title">{APP_NAME}</div>',
        unsafe_allow_html=True,
    )
    st.error(
        "No vector datasets could be loaded. Check that the GeoPackage files "
        "exist and contain valid geometries."
    )

    for year, error in vector_errors.items():
        st.write(f"**{year}:** {error}")

    st.stop()


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown(
        f'<div class="brand-title">{APP_NAME}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="brand-subtitle">{APP_SUBTITLE}</div>',
        unsafe_allow_html=True,
    )

with header_right:
    selected_year = st.selectbox(
        "Prediction year",
        available_years,
        index=len(available_years) - 1,
        key="selected_year",
    )

selected_gdf = vectors_by_year[selected_year]
selected_area = selected_gdf["area_ha"].dropna()
total_area_ha = float(selected_area.sum()) if len(selected_area) else 0.0
field_count = len(selected_gdf)
mean_area_ha = float(selected_area.mean()) if len(selected_area) else 0.0

st.divider()


# ============================================================
# MAP PAGE
# ============================================================

if page == "Map":
    map_column, stats_column = st.columns(
        [2.15, 1],
        gap="large",
    )

    with map_column:
        st.markdown(
            '<div class="section-title">Lavender field map</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="muted">Prediction layer · {selected_year}</div>',
            unsafe_allow_html=True,
        )

        rgba, raster_bounds, raster_error = raster_to_overlay(
            str(RASTER_PATH)
        )

        if raster_error:
            st.warning(raster_error)

        # Use the raster extent as the initial map extent if available.
        if raster_bounds:
            south, west = raster_bounds[0]
            north, east = raster_bounds[1]
            center = [(south + north) / 2, (west + east) / 2]
            map_obj = folium.Map(
                location=center,
                zoom_start=9,
                tiles="OpenStreetMap",
                control_scale=True,
                prefer_canvas=True,
            )

            # Transparent pixels show the map below them.
            # To make NoData areas blank instead, use tiles=None below.
            if rgba is not None:
                folium.raster_layers.ImageOverlay(
                    image=rgba,
                    bounds=raster_bounds,
                    opacity=1.0,
                    name="Sentinel-2 image",
                    interactive=False,
                    cross_origin=False,
                    zindex=1,
                ).add_to(map_obj)

        else:
            map_obj = folium.Map(
                location=[43.5, 27.5],
                zoom_start=8,
                tiles="OpenStreetMap",
                control_scale=True,
                prefer_canvas=True,
            )

        # Reproject polygons to geographic coordinates for Folium.
        map_gdf = selected_gdf.to_crs("EPSG:4326")

        folium.GeoJson(
            map_gdf,
            name=f"Lavender fields {selected_year}",
            style_function=lambda feature: {
                "color": POLYGON_COLOR,
                "weight": 1.4,
                "fillColor": POLYGON_FILL,
                "fillOpacity": POLYGON_FILL_OPACITY,
            },
            highlight_function=lambda feature: {
                "color": "#FFFFFF",
                "weight": 2.5,
                "fillOpacity": 0.65,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["area_ha"],
                aliases=["Field area (ha):"],
                localize=True,
                sticky=False,
            ),
        ).add_to(map_obj)

        # Fit to prediction polygons so they remain visible.
        try:
            bounds = map_gdf.total_bounds
            if np.all(np.isfinite(bounds)):
                map_obj.fit_bounds(
                    [
                        [bounds[1], bounds[0]],
                        [bounds[3], bounds[2]],
                    ],
                    padding=(20, 20),
                )
        except Exception:
            pass

        folium.LayerControl(collapsed=True).add_to(map_obj)

        st_folium(
            map_obj,
            use_container_width=True,
            height=610,
            returned_objects=[],
            key=f"map_{selected_year}",
        )

        st.caption(
            "Purple polygons show detected lavender fields. "
            "Use the map controls to zoom and explore individual fields."
        )

    with stats_column:
        st.markdown(
            '<div class="section-title">Overview</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Detection results for {selected_year}")

        st.metric(
            "Mapped lavender area",
            f"{total_area_ha:,.1f} ha",
        )

        st.metric(
            "Detected fields",
            f"{field_count:,}",
        )

        st.metric(
            "Mean field size",
            f"{mean_area_ha:.2f} ha",
        )

        st.divider()

        st.markdown(
            '<div class="section-title">Mapped area by year</div>',
            unsafe_allow_html=True,
        )

        yearly_rows = []
        for year, gdf in vectors_by_year.items():
            areas = gdf["area_ha"].dropna()
            yearly_rows.append(
                {
                    "Year": str(year),
                    "Mapped area (ha)": float(areas.sum()) if len(areas) else 0.0,
                }
            )

        yearly_df = pd.DataFrame(yearly_rows).set_index("Year")

        st.bar_chart(
            yearly_df,
            y="Mapped area (ha)",
            height=220,
        )

        st.divider()

        st.markdown(
            '<div class="section-title">Field size distribution</div>',
            unsafe_allow_html=True,
        )

        if len(selected_area):
            bins = [0, 1, 2, 5, 10, 20, np.inf]
            labels = [
                "0–1",
                "1–2",
                "2–5",
                "5–10",
                "10–20",
                "20+",
            ]

            size_groups = pd.cut(
                selected_area,
                bins=bins,
                labels=labels,
                right=False,
                include_lowest=True,
            )

            histogram_df = (
                size_groups.value_counts(sort=False)
                .rename_axis("Field size (ha)")
                .to_frame("Fields")
            )

            st.bar_chart(
                histogram_df,
                y="Fields",
                height=220,
            )
        else:
            st.info("No valid field-area measurements are available.")


# ============================================================
# STATISTICS PAGE
# ============================================================

elif page == "Statistics":
    st.subheader(f"Detection statistics — {selected_year}")

    c1, c2, c3 = st.columns(3)

    c1.metric("Mapped area", f"{total_area_ha:,.1f} ha")
    c2.metric("Detected fields", f"{field_count:,}")
    c3.metric("Mean field size", f"{mean_area_ha:.2f} ha")

    st.divider()

    if len(selected_area):
        stat_table = pd.DataFrame(
            {
                "Metric": [
                    "Field count",
                    "Total mapped area (ha)",
                    "Mean field area (ha)",
                    "Median field area (ha)",
                    "Minimum field area (ha)",
                    "Maximum field area (ha)",
                    "Standard deviation (ha)",
                ],
                "Value": [
                    len(selected_area),
                    selected_area.sum(),
                    selected_area.mean(),
                    selected_area.median(),
                    selected_area.min(),
                    selected_area.max(),
                    selected_area.std(),
                ],
            }
        )

        st.dataframe(
            stat_table,
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("Field size distribution")

        fig, ax = plt.subplots()
        ax.hist(selected_area, bins=20)
        ax.set_xlabel("Field area (ha)")
        ax.set_ylabel("Number of fields")
        ax.set_title(f"Field-size histogram — {selected_year}")
        ax.grid(axis="y", alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info("No valid field-area measurements are available.")


# ============================================================
# YEAR COMPARISON PAGE
# ============================================================

elif page == "Comparison":
    st.subheader("Year-to-year comparison")

    comparison_rows = []

    for year, gdf in sorted(vectors_by_year.items()):
        areas = gdf["area_ha"].dropna()

        comparison_rows.append(
            {
                "Year": year,
                "Detected fields": len(gdf),
                "Mapped area (ha)": float(areas.sum()) if len(areas) else 0.0,
                "Mean field area (ha)": float(areas.mean()) if len(areas) else 0.0,
                "Median field area (ha)": float(areas.median()) if len(areas) else 0.0,
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)

    st.dataframe(
        comparison_df,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Mapped area by year")

    area_chart = comparison_df.set_index("Year")[["Mapped area (ha)"]]
    st.bar_chart(area_chart)

    st.subheader("Detected field count by year")

    count_chart = comparison_df.set_index("Year")[["Detected fields"]]
    st.bar_chart(count_chart)

    st.caption(
        "These figures summarise each year's GeoPackage. "
        "They do not by themselves measure classification accuracy "
        "or prove that changes between years are real."
    )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "About":
    st.subheader("About LAVEO")

    st.markdown(
        """
        **LAVEO** — Lavender Area Vectorisation with Earth Observation — is a
        geospatial AI project for mapping lavender fields from Sentinel-2
        satellite imagery.

        ### Workflow

        1. Prepare Sentinel-2 imagery.
        2. Run the ResUNet segmentation model.
        3. Convert predicted pixels into field polygons.
        4. Apply vector post-processing and area filtering.
        5. Explore mapped fields and summary statistics in this dashboard.

        ### Study area

        Dobrich Province, Bulgaria.

        ### Technologies

        - Python
        - PyTorch / ResUNet
        - Sentinel-2
        - Rasterio
        - GeoPandas and Shapely
        - Folium and Streamlit

        ### Important note

        Mapped area and field counts are derived from the provided prediction
        polygons. They are not independent validation metrics. Accuracy
        requires comparison against suitable reference data.
        """
    )

    st.divider()

    st.markdown("### Input files")

    st.write(f"**Sentinel-2 raster:** `{RASTER_PATH.relative_to(ROOT)}`")

    for year, path in VECTOR_PATHS.items():
        status = "Found" if path.exists() else "Missing"
        st.write(f"**{year}:** `{path.relative_to(ROOT)}` — {status}")


# ============================================================
# DATASET WARNINGS
# ============================================================

if vector_errors:
    with st.expander("Data loading warnings"):
        for year, error in vector_errors.items():
            st.warning(f"{year}: {error}")

st.divider()

st.caption(
    "LAVEO · Lavender mapping from Earth observation · "
    "Statistics are calculated from the loaded prediction datasets."
)
