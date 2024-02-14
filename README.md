# Delimitation of the free areas on rooftops

Goal: Determine the space available on rooftops by detecting objects. Production of a binary (free/occupied) vector layer.

**Table of content**

- [Requirements](#requirements)
	- [Hardware](#hardware)
    - [Installation](#installation)
- [Classification of occupancy](#classification-of-the-roof-plane-occupation)
- [LiDAR segmentation](#lidar-segmentation)
- [Image segmentation](#image-segmentation)

## Requirements

### Hardware

For the processing of the *images*, a CUDA-compatible GPU is needed. <br>
For the processing of the *LiDAR point cloud*, there is no hardware requirement.

### Installation

The scripts have been developed with Python 3.8<!-- 3.10 actually for the pcdseg -->. For the image processing, PyTorch version 1.10 and CUDA version 11.3 were used.

All the dependencies required for the project are listed in `requirements.in` and `requirements.txt`. To install them:

- Create a Python virtual environment

        python3 -m venv <dir_path>/[name of the virtual environment]
        source <dir_path>/[name of the virtual environment]/bin/activate

- Install dependencies

        pip install -r requirements/requirements.txt

    - Requirements for the image workflow only

            pip install -r requirements/requirements_images.txt

    - Requirements for the LiDAR workflow only

            pip install -r requirements/requirements_lidar.txt


_requirements.txt_ has been obtained by compiling _requirements.in_. Recompiling the file might lead to libraries version changes:

        pip-compile requirements.in

The library `segment-geospatial` is used in "editable" mode. The modified version can be clone from this forked repository: https://github.com/swiss-territorial-data-lab/segment-geospatial.git. To install it in your virtual environment execute the following commands:

```
cd segment-geospatial
git checkout ch/dev
pip install 
```

or  in editable mode

```
pip install -e .
```

If the installation is successful the message "You are using a modified version of segment-geospatial library (v 0.10.2 fork)" must be printed in the prompt while executing the script `segment_images.py`.  

**Disclaimer**: We do not guaranty that the scripts in the sandbox folder and outside the main proposed workflows are functional in the proposed virtual environment.

## Classification of the roof plane occupation

## LiDAR segmentation

### Data

- LiDAR point cloud. Here, the [tiles of the 2019 flight over the Geneva canton](https://ge.ch/sitggeoportal1/apps/webappviewer/index.html?id=311e4a8ae2724f9698c9bcfb6ab45c56) were used.
- Delimitation of the roof for each EGID.

This workflow was tested with a ground truth produced for this project. The ground truth is available ... <br>
The ground truth is split into the training and test set to see if the algorithm performs equally on new buildings.

### Workflow

First, the segmentation is performed with different parameters on buildings with a pitched roof than on other buildings.

```
python scripts/pcd_segmentation/prepare_data.py config/config_pcdseg_all_roofs.yaml
python scripts/pcd_segmentation/pcd_segmentation.py config/config_pcdseg_all_roofs.yaml
python scripts/pcd_segmentation/vectorization.py config/config_pcdseg_all_roofs.yaml
python scripts/pcd_segmentation/prepare_data.py config/config_pcdseg_pitched_roofs.yaml
python scripts/pcd_segmentation/pcd_segmentation.py config/config_pcdseg_pitched_roofs.yaml
python scripts/pcd_segmentation/vectorization.py config/config_pcdseg_pitched_roofs.yaml
```

Then, the results for the pitched roofs and the general results are merged. Their geometry is also simplified with a buffer and copping operation, as well as with the Visvalingam-Wyatt algorithm. The different obstacles are merged together to form the occupied surfaces.

```
python scripts/pcd_segmentation/post_processing.py config/config_pcdseg_all_roofs.yaml
```

Finally, the results are assessed

```
python scripts/assessment/assess_results.py config/config_pcdseg_all_roofs.yaml
python scripts/assessment/assess_area.py config/config_pcdseg_all_roofs.yaml
```

More in details, the scripts used above perform the following steps:
1. `prepare_data.py`: reads and filter the 3D point cloud data to keep the roofs of the selected EGIDs,
2. `pcd_segmentation.py`: segments in planes and clusters the point cloud data,
3. `vectorization.py`: creates 2D polygons from the segmented point cloud data,
7. `post_processing.py`: merges the results for the pitched and general roofs together and simplify the geometry of the detections.
5. `assess_results.py`: evaluate the results based on the ground truth,
6. `assess_area.py`: calculates the free and occupied surface of each EGIDs and compare it with the ground truth.

The workflow described here is working with the training subset of the ground truth used for the optimization of the hyperparameters. The configuration file `config_pcdseg_test.yaml` works with the test subset of the ground truth, allowing to test on buildings not considered in the optimization.

The optimization of hyperparameters can be performed as follow:

```
python scripts/pcd_segmentation/optimize_hyperparameters.py config/config_pcdseg_all_roofs.yaml
```

## Image segmentation

### Overview
The set of scripts is dedicated to object detection in images. Tiles fitting to the extension of buildings in a given AOI are produced. Images are segmented using [segment-geospatial](https://github.com/opengeos/segment-geospatial) which provides a practical framework to using [SAM](https://github.com/facebookresearch/segment-anything) (**Segment-Anything Model**) with georeferenced data. Detection masks are converted to vectors and filtered. Finally, the results are evaluated by comparing them with Ground Truth labels defined by domain experts. 

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

    python3 scripts/image_segmentation/generate_tiles.py config/config_imgseg.yaml
    python3 scripts/image_segmentation/segment_images.py config/config_imgseg.yaml
    python3 scripts/image_segmentation/produce_vector_layer.py config/config_imgseg.yaml
    python3 scripts/assessment/assess_results.py config/config_imgseg.yaml
    python3 scripts/assessment/assess_surface.py config/config_imgseg.yaml

The model optimization can be performed as follow:

    python3 scripts/image_segmentation/optimize_hyperparameters.py config/config_imgseg.yaml
