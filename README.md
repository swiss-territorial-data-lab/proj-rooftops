# proj-rooftops

**Aim**: Determine the space availbale on rooftops by detecting objects. Production of a binary (free/occupied) vector layer for beneficiaries.

## Image segmentation 

### Overview
The set of scripts are dedicated to detecting objects in images. Tiles fitting the extension of building in a given AOI are produced. The **Segment-Anything Model** (https://github.com/facebookresearch/segment-anything) is then used to perform image segmentation using a pre-trained model. The detection masks are converted to vectors and filtered. Finally, the results are evaluated by comparing them with labels defined by domain experts. To process SAM with georeferenced data, the framework `segment-geospatial` (https://github.com/opengeos/segment-geospatial) is used. 

### Requirements

#### Hardware

The scripts have been run with Ubuntu 20.04 OS on a 32 GiB RAM machine with 16 GiB GPU (NVIDIA Tesla T4)

#### Installation

The scripts have been developed with Python 3.8 using PyTorch version 1.10 and CUDA version 11.3.

All the dependencies required for the project are listed in `requirements.in` and `requirements.txt`. To install them:

- Create a Python virtual environment

        $ python3 -m venv <dir_path>/[name of the virtual environment]
        $ source <dir_path>/[name of the virtual environment]/bin/activate

- Install dependencies

        $ pip install -r requirements.txt

-_requirements.txt_ has been obtained by compiling _requirements.in_. Recompiling the file might lead to libraries version changes:

        $ pip-compile requirements.in


### Files structure

1. `generate_tiles.py` : produces custom tiles of the extension of a given roof
2. `segment_images.py` : image segmentation: detection masks + masks vectorisation
3. `produce_vector_layer.py` : filtering of the vector layer for each building + aggregation of all layers into a single layer (binary: free/occupied)
4. `assess_results.py` : evaluation of the results. Comparison of results to Ground Truth, metrics computation and detection tagging.
5. `optimize_hyperparameters.py` : Optuna framework to optimize hyperparameters

### Data

This part of the project uses True Orthophotos processed from flight acquisitions of the Geneva canton in 2019.

- True Orthophotos (image_dir): /mnt/s3/proj-rooftops/02_Data/initial/Geneva/ORTHOPHOTOS/2019/TIFF_TRUEORTHO/*.tiff

Shapefiles are also used as input data and listed below:

- Data linked to the building selection of the Ground Truth:

    - Roof shape (roofs): shapefile derived from the layer CAD_BATIMENT_HORSOL_TOIT.shp (https://ge.ch/sitg/sitg_catalog/sitg_donnees?keyword=&geodataid=0635&topic=tous&service=tous&datatype=tous&distribution=tous&sort=auto) filter with the selected EGID of buildings: /mnt/s3/proj-rooftops/02_Data/ground_truth/PanData/egid/CAD_BATIMENT_HORSOL_TOIT_GTselection_v3.shp
    - Tiles shape (tiles): shapefile of the True Orthophotos tiles overlapping the selected buidlings: /mnt/s3/proj-rooftops/02_Data/ground_truth/PanData/tiles/Tiles_GTselection_v3.shp
    - Ground truth shape (labels): shapefile of the True Orthophotos tiles overlapping the selected buidlings: /mnt/s3/proj-rooftops/02_Data/ground_truth/occupation/PanData/roofs_31_08_STDL/roofs_31_08_STDL_corrected.shp

### Workflow

Following the end-to-end, the workflow can be run by issuing the following list of actions and commands:

    $ cd proj-rooftops
    $ python3 scripts/image_processing/generate_tiles.py config/config_imgseg.yaml
    $ python3 scripts/image_processing/image_segmentation.py config/config_imgseg.yaml
    $ python3 scripts/image_processing/produce_vector_layer.py config/config_imgseg.yaml
    $ python3 scripts/image_processing/assess_results.py config/config_imgseg.yaml

The model optimization (find the hyperparameters maximizing the f1 score) can be performed as follow:

    $ python3 scripts/image_processing/optimize_hyperparameters.py config/config_imgseg.yaml