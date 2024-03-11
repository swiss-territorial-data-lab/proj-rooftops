# Detection of free and occupied surfaces on rooftops

The set of provided scripts aim to evaluate the surface available on rooftops by detecting objects. 

**Table of content**

- [Requirements](#requirements)
    - [Hardware](#hardware)
    - [Installation](#installation)
    - [General data](#general-data)
- [Classification of occupancy](#classification-of-the-roof-plane-occupation)
- [LiDAR segmentation](#lidar-segmentation)
- [Image segmentation](#image-segmentation)

## Requirements

### Hardware

Image processing was run on a 32 GiB RAM machine with 16 GiB GPU (NVIDIA Tesla T4) compatible with CUDA. <br>
There is no hardware requirement to process the LiDAR point cloud.

### Installation

The scripts were developed with Python 3.8 on Ubuntu 20.04 OS.

All the dependencies required for the project are listed in `requirements.in` and `requirements.txt`. To install them:

- Create a Python virtual environment

        python3 -m venv <dir_path>/[name of the virtual environment]
        source <dir_path>/[name of the virtual environment]/bin/activate

- Install dependencies with pip >= 20.3:

        pip install -r requirements.txt

- _requirements.txt_ has been obtained by compiling _requirements.in_. Recompiling the file might lead to libraries version changes:

        pip install -r requirements.txt

    - Requirements for the image workflow only

            pip install -r requirements_images.txt

    - Requirements for the LiDAR workflow only

            pip install -r requirements_lidar.txt
            
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
        pip install -e .
        ```

        If the installation is successful, the message "You are using a modified version of segment-geospatial library (v 0.10.2 fork)" must be printed in the prompt while executing the script `segment_images.py`.  

**Disclaimer**: We do not guarantee that the scripts in the sandbox folder and the scripts not included in the workflows can be executed with the provided requirements

### General data

The datasets are described here after:

- Roof delimitation: vector shapefile [CAD_BATIMENT_HORSOL_TOIT](https://ge.ch/sitg/sitg_catalog/sitg_donnees?keyword=&geodataid=0635&topic=tous&service=tous&datatype=tous&distribution=tous&sort=auto) providing the roof planes by EGID;
- Ground truth of the roof objects: vector shapefile of the labels produced for this project and used for the assessments of the segmentation workflows:
   - version 2023-11-10 LiDAR: ground truth used for the optimization and the assessment of the LiDAR segmentation. Most objects in the low height classes, such as lawn and terraces, have been removed from the dataset;
   - version 2023-11-13: ground truth used for the optimization and the assessment of the image segmentation. It corresponds to the complete ground truth;
- EGID lists: selected buildings for the ground truth identified by their federal number (EGID). The buildings are listed in `EGID_GT_full.csv` and split into the training and test datasets:
    - EGID_GT_test.csv: 17 EGID selected to control the performance of the algorithm on a test dataset. It is provided here as an example to run the code with.

In this repository, only test data is supplied, along with a subset of the roof delimitations, to enable the user to run an example. The full datasets can be requested by contacting the STDL.

## Classification of the roof plane occupancy

## LiDAR segmentation

The set of scripts is dedicated to the segmentation of rooftop objects in the LiDAR point cloud. This workflow is based on [Open3D](https://www.open3d.org/docs/release/). It supposes that roofs composed of flat planes and that obstacles protrude.

### Data

In addition to the general data, the segmentation workflow needs:

- LiDAR point clouds: the [tiles of the 2019 flight over the Geneva canton](https://ge.ch/sitggeoportal1/apps/webappviewer/index.html?id=311e4a8ae2724f9698c9bcfb6ab45c56) were used. They are automatically downloaded through a script;
- Emprises tiles LiDAR 2019.shp: Shapes corresponding to the LiDAR tiles.

### Workflow

The following scripts are used to segment the LiDAR point cloud:
1. `retrieve_point_clouds.py`: downloads the point clouds,
2. `prepare_data.py`: reads and filters the roofs of the selected EGID in the point clouds,
3. `pcd_segmentation.py`: segments the point clouds in planes and clusters,
4. `vectorization.py`: creates 2D polygons from the segmented point clouds,
5. `post_processing.py`: merges the results for the pitched and other roofs together and simplifies the geometry of the detections,
6. `assess_results.py`: evaluates results by comparing them with the ground truth, calculates metrics and tags detections,
7. `assess_area.py`: calculates the free and occupied surface of each EGID and compares it with the ground truth.

The corresponding command lines are provided here below.

First, the LiDAR point cloud tiles have to be downloaded with the command below and unzipped by the user.

```
python scripts/pcd_segmentation/retrieve_point_clouds.py config/config_pcdseg_all_roofs.yaml
```

After that, the point cloud segmentation is performed. Specific parameters are used for pitched roofs.

```
python scripts/pcd_segmentation/prepare_data.py config/config_pcdseg_all_roofs.yaml
python scripts/pcd_segmentation/pcd_segmentation.py config/config_pcdseg_all_roofs.yaml
python scripts/pcd_segmentation/vectorization.py config/config_pcdseg_all_roofs.yaml
python scripts/pcd_segmentation/prepare_data.py config/config_pcdseg_pitched_roofs.yaml
python scripts/pcd_segmentation/pcd_segmentation.py config/config_pcdseg_pitched_roofs.yaml
python scripts/pcd_segmentation/vectorization.py config/config_pcdseg_pitched_roofs.yaml
```

Then, the results for the pitched and other roofs are merged. Their geometry is simplified with buffering and copping operations, as well as the Visvalingam-Wyatt algorithm. The different obstacles are merged together to form the occupied surfaces.

```
python scripts/pcd_segmentation/post_processing.py config/config_pcdseg_all_roofs.yaml
```

Finally, the results are assessed

```
python scripts/assessment/assess_results.py config/config_pcdseg_all_roofs.yaml
python scripts/assessment/assess_area.py config/config_pcdseg_all_roofs.yaml
```

The workflow described here is working with the training subset of the ground truth used for the optimization of the hyperparameters. The configuration file `config_pcdseg_test.yaml` works with the test subset of the ground truth, allowing to test on buildings not considered in the optimization.

The optimization of hyperparameters can be performed as follow:

```
python scripts/pcd_segmentation/optimize_hyperparameters.py config/config_pcdseg_all_roofs.yaml
```

## Image segmentation

### Overview

The set of scripts is dedicated to the segmentation of objects in images. The segmentation is based on a deep learning method using [SAM](https://github.com/facebookresearch/segment-anything) (Segment-Anything Model). The final product is a vector layer of detected objects on the selected roofs. 

### Data

In addition to the general data, the image segmentation workflow uses the following input data:

- True orthophotos of the canton of Geneva: processed from aerial image acquired by the State of Geneva in 2019. RGB tiff images with a spatial resolution of about 7 cm/px. Images are available on request from SITG.
- Image tile shapes: vector shapefile of the true orthophoto tiles available on request from SITG.

### Script description

1. `generate_tiles.py`: generates custom tiles of the roof extent;
2. `segment_images.py`: creates detection masks and vectorizes them. Images are segmented using [segment-geospatial](https://github.com/opengeos/segment-geospatial) which provides a practical framework for using [SAM](https://github.com/facebookresearch/segment-anything) (Segment-Anything Model) with georeferenced data.
3. `produce_vector_layer.py`: filters the vector layer for each building and aggregates all layers into a single one (detected objects);
4. `assess_results.py`: evaluates results by comparing them with the ground truth, calculates metrics and tags detections;
5. `assess_area.py`: calculates the free and occupied surface of each EGID and compare it with the ground truth;
6. `optimize_hyperparameters.py`: optimizes SAM hyperparameters to maximize the desired metrics (f1-score, median IoU, precision, recall,...). Based on the [Oputna](https://optuna.org/) framework.

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
