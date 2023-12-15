# proj-rooftops

**Aim**: Determine the space available on rooftops by detecting objects. Production of a binary (free/occupied) vector layer for beneficiaries.

## Image segmentation 

### Overview
The set of scripts is dedicated to object detection in images. Tiles fitting to the extension of buildings in a given AOI are produced. Images are segmented using [segment-geospatial](https://github.com/opengeos/segment-geospatial) which provides a practical framework to using [SAM](https://github.com/facebookresearch/segment-anything) (**Segment-Anything Model**) with georeferenced data. Detection masks are converted to vectors and filtered. Finally, the results are evaluated by comparing them with Ground Truth labels defined by domain experts. 

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

-_requirements.txt_ has been obtained by compiling _requirements.in_. Recompiling the file might lead to version changes for libraries:

        $ pip-compile requirements.in

- The library `segment-geospatial` is used in "editable" mode. The modified version can be clone from this forked repository: https://github.com/swiss-territorial-data-lab/segment-geospatial.git. To install it in your virtual environment execute the following commands:

        $ cd segment-geospatial
        $ git checkout ch/dev
        $ pip install .

        or in editable mode

        $ pip install -e .

If the installation is successful the message "You are using a modified version of segment-geospatial library (v 0.10.2 fork)" must be printed in the prompt while executing the script `segment_images.py`.  

### Files description and folder structure

1. `generate_tiles.py` : produces custom tiles of the extension of a given roof,
2. `segment_images.py` : produces detection masks and vectorizes them,
3. `produce_vector_layer.py` : filters the vector layer for each building and aggregates of all layers into a single layer (binary: free/occupied),
4. `assess_results.py` : evaluates the results by comparing them to ground truth, computes metrics and and tags the detections,
5. `optimize_hyperparameters.py` : Optuna framework to optimize hyperparameters based on the f1 score and median IoU over buildings.

### Data

This part of the project uses true orthophotos processed from flight acquisitions by the Geneva canton in 2019.

- True orthophotos (image_dir): /mnt/s3/proj-rooftops/02_Data/initial/Geneva/ORTHOPHOTOS/2019/TIFF_TRUEORTHO/*.tiff

Shapefiles are also used as input data and listed below:

- Data linked to the building selection of the ground truth:

    - Roof shapes: shapefile derived from the layer [CAD_BATIMENT_HORSOL_TOIT.shp](https://ge.ch/sitg/sitg_catalog/sitg_donnees?keyword=&geodataid=0635&topic=tous&service=tous&datatype=tous&distribution=tous&sort=auto). It is filtered with the EGID of the selected buildings: /mnt/s3/proj-rooftops/02_Data/ground_truth/EGIDs_selected_GT.csv (list can be adapted)
    - Tile shape: shapefile of the true orthophoto tiles overlapping the selected buildings: /mnt/s3/proj-rooftops/02_Data/initial/Geneva/ORTHOPHOTOS/2019/TUILES_TRUEORTHO/Tuiles.shp
    - Ground truth shapes (labels): shapefile of the ground truth lables: /mnt/s3/proj-rooftops/02_Data/ground_truth/occupation/PanData/roofs_STDL_proofed_2023-11-13.shp
    - EGIDs lists (egids): /mnt/s3/proj-rooftops/02_Data/ground_truth/PanData/occupation/Partition/
        - EGIDs_GT_test.csv: list of egids selected to control the performance of the algorithm on a test dataset.
        - EGIDs_GT_training.csv: list of egids selected to perform hyperparameter optimization of algorithms on a training dataset. 
        - EGIDs_GT_training_subsample_imgseg.csv: In the case of image segmentation, the training list is too large to perform hyperparameters optimization within a reasonable time. Therefore, a reduced training list of 25 buildings is proposed. 

### Workflow

The workflow can be run by issuing the following list of actions and commands:

    $ python3 scripts/image_segmentation/generate_tiles.py config/config_imgseg.yaml
    $ python3 scripts/image_segmentation/image_segmentation.py config/config_imgseg.yaml
    $ python3 scripts/image_segmentation/produce_vector_layer.py config/config_imgseg.yaml
    $ python3 scripts/assessment/assess_results.py config/config_imgseg.yaml
    $ python3 scripts/assessment/assess_surface.py config/config_imgseg.yaml

The model optimization can be performed as follow:

    $ python3 scripts/image_segmentation/optimize_hyperparameters.py config/config_imgseg.yaml