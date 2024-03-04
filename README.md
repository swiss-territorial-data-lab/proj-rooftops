# Detection of free and occupied surfaces on rooftops

The set of provided scripts aim to evaluate the surface available on rooftops by detecting objects. 

**Table of content**

- [Requirements](#requirements)
	- [Hardware](#hardware)
    - [Installation](#installation)
- [Classification of occupancy](#classification-of-the-roof-plane-occupation)
- [LiDAR segmentation](#lidar-segmentation)
- [Image segmentation](#image-segmentation)

## Requirements

### Hardware

Image processing was run on a 32 GiB RAM machine with 16 GiB GPU (NVIDIA Tesla T4) compatible with CUDA. <br>
No hardware requirements are needed to process the LiDAR point cloud.

### Installation

The scripts were developed with Python 3.8<!-- 3.10 actually for the pcdseg --> on Unbuntu 20.04 OS.

All the dependencies required for the project are listed in `requirements.in` and `requirements.txt`. To install them:

- Create a Python virtual environment

        python3 -m venv <dir_path>/[name of the virtual environment]
        source <dir_path>/[name of the virtual environment]/bin/activate

- Install dependencies with pip >= 20.3:

        pip install -r requirements/requirements.txt

- _requirements.txt_ has been obtained by compiling _requirements.in_. Recompiling the file might lead to libraries version changes:

        pip install -r requirements/requirements.txt

    - Requirements for the image workflow only

            pip install -r requirements/requirements_images.txt

    - Requirements for the LiDAR workflow only

            pip install -r requirements/requirements_lidar.txt
            
- Specific libraries were used for image processing:
    - PyTorch version 1.10
    - CUDA version 11.3
    - segment-geospatial [0.10.2](https://github.com/opengeos/segment-geospatial/releases/tag/v0.10.2). <br>
    The library was adapted to our needs and can be cloned from this forked repository: https://github.com/swiss-territorial-data-lab/segment-geospatial.git. <br> To install it in your virtual environment, execute the following commands:

        ```
        cd segment-geospatial
        git checkout ch/dev
        pip install .
        ```

        or in editable mode:

        ```
        $ pip install -e .
        ```

        If the installation is successful, the message "You are using a modified version of segment-geospatial library (v 0.10.2 fork)" must be printed in the prompt while executing the script `segment_images.py`.  

**Disclaimer**: We do not guaranty that the scripts in the sandbox folder and outside the main proposed workflows are functional in the proposed virtual environment.

## Classification of the roof plane occupancy

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

The set of scripts is dedicated to the segmentation of objects in images. The segmentation is based on a deep learning method using [SAM](https://github.com/facebookresearch/segment-anything) (Segment-Anything Model). The final product is a vector layer of detected objects on the selected roofs. 

### Data

The image segmentation workflow uses the following input data:

- True orthophotos of the canton of Geneva: processed from aerial image acquired by the State of Geneva in 2019. RGB tiff images with a spatial resolution of about 7 cm/px. Images are available on request from SITG.
- Image tile shapes: vector shapefile of the true orthophoto tiles available on request from SITG. 
- Roof delimitation: vector shapefile [CAD_BATIMENT_HORSOL_TOIT.shp](https://ge.ch/sitg/sitg_catalog/sitg_donnees?keyword=&geodataid=0635&topic=tous&service=tous&datatype=tous&distribution=tous&sort=auto) providing roof planes by EGID. 
- Ground truth objects: vector shapefile of the ground truth labels
- EGIDs lists: lists of selected buildings (EGID = building identifier). The buildings are split into different datasets:
    - EGIDs_GT_full.csv: list of 122 EGIDs selected to establish the labels vectorization
    - EGIDs_GT_test.csv: list of 17 EGIDs selected to control the performance of the algorithm on a test dataset.
    - EGIDs_GT_training.csv: list of 105 EGIDs selected to perform hyperparameter optimization of algorithms on a training dataset. 
    - EGIDs_GT_training_subsample_imgseg.csv: In the case of image segmentation, the training list is too long to perform hyperparameters optimization in a reasonable time. For this reason, a training list reduced to 25 buildings is provided. 

### Script description

1. `generate_tiles.py`: generates custom tiles of the roof extent
2. `segment_images.py`: creates detection masks and vectorize them. Images are segmented using [segment-geospatial](https://github.com/opengeos/segment-geospatial) which provides a practical framework for using [SAM](https://github.com/facebookresearch/segment-anything) (Segment-Anything Model) with georeferenced data.
3. `produce_vector_layer.py`: filters the vector layer for each building and aggregates all layers into a single one (detected objects)
4. `assess_results.py`: evaluates results by comparing them with the ground truth, calculates metrics and tags detections
5. `optimize_hyperparameters.py`: optimizes SAM hyperparameters to maximize the desired metrics (f1-score, median IoU, precision, recall,...). Based on the [Oputna](https://optuna.org/) framework

### Workflow

The workflow can be run by issuing the following list of commands:

```
python scripts/image_segmentation/generate_tiles.py config/config_imgseg.yaml
python scripts/image_segmentation/segment_images.py config/config_imgseg.yaml
python scripts/image_segmentation/filter_merge_detections.py config/config_imgseg.yaml
python scripts/assessment/assess_results.py config/config_imgseg.yaml
python scripts/assessment/assess_area.py config/config_imgseg.yaml
```

The optimization of hyperparameters can be performed as follow:

```
python scripts/image_segmentation/optimize_hyperparameters.py config/config_imgseg.yaml
```
