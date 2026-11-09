# LAVEO: Lavender Area Vectorisation with Earth Observation

LAVEO is a geospatial AI project for mapping lavender fields from Sentinel-2 satellite imagery using deep learning and geospatial post-processing.

The project uses a ResUNet semantic segmentation model to identify lavender fields, followed by raster-to-vector conversion and geometry processing to produce georeferenced field boundaries.

![LAVEO lavender field predictions over Sentinel-2 imagery](docs/images/vis.png)

## 1. Workflow

The processing workflow consists of six main stages:

1. **Satellite imagery acquisition** – Downloads Sentinel-2 imagery for the training, test and prediction areas.
2. **Data preparation** – Prepares imagery, labels and spatial patches for model development and inference.
3. **Input preparation** – Organises image tiles and input data for the machine learning workflow.
4. **Deep learning segmentation** – Trains and evaluates a ResUNet model using labelled lavender field data.
5. **Full-tile inference** – Applies the trained model to satellite imagery to generate lavender prediction rasters.
6. **Geospatial post-processing** – Converts raster predictions into polygons, removes small polygons and simplifies boundaries while preserving topology.

The workflow is orchestrated using Dagster.

## 2. Study Area and Data

**Study area:** Dobrich Province, Bulgaria.

### Input datasets

The repository includes:

- Training lavender field boundaries (2021, 2023 and 2025).
- Test lavender field boundaries (2021, 2023 and 2025).
- Training Area of Interest (AOI).
- Test Area of Interest (AOI).

These datasets are stored in:

```text
data/
├── labels/
│   ├── Train_Bg_Lavender fields-07-2021.geojson
│   ├── Train_Bg_Lavender fields-07-2023.geojson
│   ├── Train_Bg_Lavender fields-07-2025.geojson
│   ├── Test_Bg_Lavender fields-07-2021.geojson
│   ├── Test_Bg_Lavender fields-07-2023.geojson
│   └── Test_Bg_Lavender fields-07-2025.geojson
└── training&val_aoi/
    ├── Bulgaria_Dobrich_training_AOI.geojson
    └── Bulgaria_Dobrich_test_AOI.geojson
```

Satellite imagery and intermediate processing products are generated during pipeline execution and are not included in the repository.

## 3. Technology

- Python 3.11
- PyTorch
- NumPy and Pandas
- Rasterio
- GeoPandas
- Shapely
- PySTAC and ODC-STAC
- Xarray and Rioxarray
- Dagster
- Streamlit
- Folium
- Sentinel-2 satellite imagery

## 4. Installation

### Requirements

- Windows, Linux or macOS.
- Miniconda or Anaconda.
- Git.
- Internet connection for satellite imagery acquisition and dependency installation.

### Clone the repository

```bash
git clone https://github.com/unagaev/laveo.git
cd laveo
```

### Create the Conda environment

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate laveo
```

### Install the project as a package

```bash
pip install -e .
```

Editable installation registers the `lavender` package in the active Python environment, allowing project modules to be imported without manually configuring PYTHONPATH.

### Verify the installation

```bash
python --version
dagster --version
streamlit --version
python -c "import torch, rasterio, geopandas, lavender; print('Dependencies OK')"
```

## 5. Running the project

There are three ways to interact with the project.

### Option A: Run the processing scripts individually

Individual Python scripts can be executed directly from the project root.

This approach is useful for:

- Development and debugging.
- Testing individual processing stages.
- Inspecting intermediate outputs.
- Re-running selected processing steps.

The main processing scripts are located in `src/lavender/`:

| Step | Script | Purpose |
|---|---|---|
| 1 | `ingestion/download_training_test_tiles.py` | Downloads training and test imagery |
| 2 | `ingestion/download_prediction_tiles.py` | Downloads prediction imagery |
| 3 | `utils/rasterise_tile.py` | Rasterises field boundaries |
| 4 | `utils/patches_tiles_all_years.py` | Generates spatial patches |
| 5 | `utils/convert_patches_to_npy_multiyear.py` | Converts patches to NumPy arrays |
| 6 | `ml/training/train.py` | Trains the ResUNet model |
| 7 | `ml/eval/evaluate.py` | Evaluates model performance |
| 8 | `ml/inference/predict_tile.py` | Generates full-tile predictions |
| 9 | `postprocessing/postprocessing.py` | Extracts and processes field polygons |

The scripts should be executed in the appropriate processing order. Required input data, configurations and output directories must be available before running each script.

### Option B: Run the complete workflow using Dagster (recommended)

Dagster manages the processing workflow through assets and their dependencies.

From the project root, start the Dagster development server:

```bash
dagster dev -m orchestration.definitions
```

Open the Dagster interface at:

```text
http://localhost:3000
```

The project defines assets for:

1. Downloading training and prediction tiles.
2. Preparing training and prediction tiles.
3. Preparing model input tiles.
4. Training and evaluating the model.
5. Generating full-tile predictions.
6. Post-processing the predictions.

In the Dagster interface, inspect the asset graph and materialise the required assets.

**Note:** Loading and validating Dagster definitions only checks that the workflow definitions can be loaded. It does not execute the complete processing pipeline.

### Option C: Run the Streamlit visualisation

The project includes a Streamlit application for visualising geospatial predictions.

From the project root:

```bash
streamlit run frontend/app.py
```

The application opens in a browser, normally at:

```text
http://localhost:8501
```

The visualisation requires the relevant prediction and geospatial output files to have been generated.

## 6. Project structure

```text
LAVEO/
├── data/
│   ├── labels/
│   └── training&val_aoi/
│
├── docs/
│   ├── images/
│   └── vis.png
│
├── frontend/
│   └── app.py
│
├── orchestration/
│   ├── __init__.py
│   └── definitions.py
│
├── src/
│   └── lavender/
│       ├── __init__.py
│       ├── ingestion/
│       ├── ml/
│       │   ├── dataset/
│       │   ├── eval/
│       │   ├── inference/
│       │   ├── models/
│       │   └── training/
│       ├── postprocessing/
│       └── utils/
│
├── .gitignore
├── config.toml
├── environment.yml
├── LICENSE
├── pyproject.toml
└── README.md
```

## 7. Outputs

The project produces:

- Georeferenced raster predictions.
- Vector polygon datasets representing detected lavender fields.
- Model evaluation metrics.
- Intermediate image patches and processing products.

The vector outputs represent predicted lavender field boundaries in Dobrich Province, Bulgaria.

Generated data, model checkpoints and intermediate files are excluded from GitHub to keep the repository manageable.

## 8. Limitations

1. **Single-date imagery:** Current predictions use predefined single-date Sentinel-2 imagery. Improved cloud masking and multi-date compositing could help reduce the impact of cloud cover and variable image conditions.

2. **Prediction accuracy:** Model performance depends on image quality, spatial resolution and the accuracy and representativeness of training labels. Further model development and validation are needed to improve prediction reliability.

3. **Geographical transferability:** The current implementation focuses on Dobrich Province, Bulgaria. Its ability to generalise to other regions and countries requires further evaluation.

4. **Boundary delineation:** Predicted field boundaries may contain geometric inaccuracies, fragmented polygons or irregular edges. Further post-processing and comparison against reference boundaries would help assess and improve their quality.

## 9. Project status

**Research and development.**

The workflow is being developed and evaluated for lavender field detection, geospatial boundary extraction and cross-region transferability.

## 10. License

See the [LICENSE](LICENSE) file for licensing information.