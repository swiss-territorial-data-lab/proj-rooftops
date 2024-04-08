# Detection of free and occupied surfaces on rooftops

The set of provided scripts aim to evaluate the surface available on rooftops by detecting objects. 

**Table of content**

- [Requirements](#requirements)
    - [Hardware](#hardware)
    - [Installation](#installation)
    - [General data](#general-data)
- [Classification of occupancy](#classification-of-the-roof-plane-occupancy)
- [LiDAR segmentation](#lidar-segmentation)
- [Image segmentation](#image-segmentation)
- [Combination of segmentation results](#combination-of-segmentation-results)

## Requirements

### Hardware

Image processing was run on a 32 GiB RAM machine with 16 GiB GPU (NVIDIA Tesla T4) compatible with CUDA. <br>
There is no hardware requirement to process the LiDAR point cloud.

### Installation

The scripts were developed with Python 3.8 on Ubuntu 20.04 OS.

All the dependencies required for the project are listed in `requirements.in` and `requirements.txt`. To install them:

- Create a Python virtual environment

        python3 -m venv <dir_path>/<name of the virtual environment>
        source <dir_path>/<name of the virtual environment>/bin/activate

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

        $ cd segment-geospatial
        $ git checkout ch/dev
        $ pip install .

        or in editable mode

        $ pip install -e .

If the installation is successful the message "You are using a modified version of segment-geospatial library (v 0.10.2 fork)" must be printed in the prompt while executing the script `segment_images.py`.  

**Disclaimer**: We do not guarantee that the scripts in the sandbox folder and the scripts not included in the workflows can be executed with the provided requirements


### Data

The datasets needed for **all workflows** are described here after:

- Roof delimitation: vector shapefile [CAD_BATIMENT_HORSOL_TOIT](https://ge.ch/sitg/sitg_catalog/sitg_donnees?keyword=&geodataid=0635&topic=tous&service=tous&datatype=tous&distribution=tous&sort=auto) providing the roof planes by EGID;
- Ground truth of the roof objects: vector shapefile of the labels produced for this project and used for the assessments of the segmentation workflows:
   - version 2023-11-10 LiDAR: ground truth used for the optimization and the assessment of the LiDAR segmentation. Most objects in the low height classes, such as lawn and terraces, have been removed from the dataset;
   - version 2023-11-13: ground truth used for the optimization and the assessment of the image segmentation. It corresponds to the complete ground truth;
- EGID lists: selected buildings for the ground truth identified by their federal number (EGID). The buildings are listed in `EGID_GT_full.csv` and split into the training and test datasets:
    - EGID_GT_test.csv: 17 EGIDs selected to control the performance of the algorithm on a test dataset. It is provided here as an example to run the code with.

In addition, the workflows working with **LiDAR** need:

- LiDAR point clouds: the [tiles of the 2019 flight over the Geneva canton](https://ge.ch/sitggeoportal1/apps/webappviewer/index.html?id=311e4a8ae2724f9698c9bcfb6ab45c56) were used. They are automatically downloaded through a script;
- Emprises tiles LiDAR 2019.shp: shapes corresponding to the LiDAR tiles.

In addition, the workflow for the **image** segmentation needs:

- True orthophotos of the canton of Geneva: processed from aerial image acquired by the State of Geneva in 2019. RGB tiff images with a spatial resolution of about 7 cm/px. Images are available on request from SITG.
- Image tile shapes: vector shapefile of the true orthophoto tiles available on request from SITG.


In this repository, only test data is supplied, along with a subset of the roof delimitations, to enable the user to run an example. The full datasets can be requested by contacting the STDL.


## Classification of the roof plane occupancy

**Goal**: Classify the roof planes as "occupied" or "potentially free" based on their roughness and intensity.

## Workflow

(*facultative*) The script `get_lidar_infos.py` allows to get some characteristics of the point clouds.

The following scripts are used to classify roof planes by occupancy:
1. `rasterize_intensity.py`: creates an intensity raster for each LiDAR point cloud in the input directory.
    - The parameters and the function for the raster of intensity are referenced here: [LidarIdwInterpolation - WhiteboxTools](https://www.whiteboxgeo.com/manual/wbt_book/available_tools/lidar_tools.html#LidarIdwInterpolation)
2. `rasterize_roughness.py`: creates a DEM and saves it in a raster, then estimates the multi-scale roughness from the DEM.
    - The parameters and the function for the DEM are referenced here: [LidarDigitalSurfaceModel - WhiteboxTools](https://www.whiteboxgeo.com/manual/wbt_book/available_tools/lidar_tools.html#LidarDigitalSurfaceModel)
    - The parameters and the function for the multi-scale roughness are referenced here: [MultiscaleRoughness - WhiteboxTools](https://www.whiteboxgeo.com/manual/wbt_book/available_tools/geomorphometric_analysis.html#MultiscaleRoughness)
3. `get_zonal_stats.py`: gets zonal stats of intensity and roughness for roof planes.
    - Roof planes smaller than 2 m<sup>2</sup> are classified as "occupied" and no zonal stats are calculated. They are too small for a solar or vegetated installation.
    - When LiDAR point cloud is not classified as building under roof planes, they are classified as "undefined". The existence of a roof at this location should be controlled.
4. Two possibilities were developed for classification:
    1. Using manual thresholds *without ground truth*:
        - `manual_thresholds.py`: classifies the roofs using threshold passed in the config file.
        - `assess_classif_surfaces.py`: if some ground truth is provided later on or an expert assess the result, calculates the precision of the classification, also called "satisfaction  rate" in the documentation.
    2. Using a random forest *with the ground truth* 
        - `random_forest.py`:
            - train mode: if the parameter `TRAIN` is set to `True`, trains a model per office and saves them as pickle files, assesses the quality of the classification.
            - inference mode: if the parameter `TRAIN` is set to `False`, uses the saved models to make inferences on the roof planes.

The corresponding command lines are provided here below.

```
python scripts/occupation_classification/rasterize_intensity.py config/config_occupation_classification.yaml
python scripts/occupation_classification/rasterize_roughness.py config/config_occupation_classification.yaml
python scripts/occupation_classification/get_zonal_stats.py config/config_occupation_classification.yaml
```

When *no ground truth is available*, the classification can be performed with the script `manual_thresholds.py` using thresholds calibrated manually by an operator. The results can then eventually be assessed by experts, their quality assessed, and used as ground truth.

```
python scripts/occupation_classification/manual_thresholds.py config/config_occupation_classification.yaml
python scripts/assessment/assess_classif_surfaces.py config/config_occupation_classification.yaml
```

When *a ground truth is available*, the classification can be performed and assessed with the script `random_forest.py`.

```
python scripts/occupation_classification/random_forest.py config/config_occupation_classification.yaml
```

Other scripts are present in the folder `scripts/occupation_classification`. Their goal is to detect objects based on intensity. The results were not as good as expected and they were therefore not implemented in the final workflow.


## LiDAR segmentation

The set of scripts is dedicated to the segmentation of rooftop objects in the LiDAR point cloud. This workflow is based on [Open3D](https://www.open3d.org/docs/release/). It supposes that roofs composed of flat planes and that obstacles protrude.

### Workflow

The following scripts are used to segment the LiDAR point cloud:
1. `retrieve_point_clouds.py`: downloads the point clouds,
2. `prepare_data.py`: reads and filters the roofs of the selected EGIDs in the point clouds,
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
python scripts/pcd_segmentation/optimize_hyperparam_LiDAR.py config/config_pcdseg_all_roofs.yaml
```


## Image segmentation

### Overview

The set of scripts is dedicated to the segmentation of objects in images. The segmentation is based on a deep learning method using [SAM](https://github.com/facebookresearch/segment-anything) (Segment-Anything Model). The final product is a vector layer of detected objects on the selected roofs. 

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

## Combination of segmentation results

### Overview

LiDAR and image segmentation results can be combined. Two methods are used:
- Polygon concatenation: the detection polygons obtained from LiDAR segmentation and image segmentation are concatenated. 
- Polygon filtering with spatial join: the detection polygons obtained from image segmentation are filtered, retaining only polygons that overlap those obtained from LiDAR segmentation.

### Script description

1. `combine_results_seg.py`: combines results of LiDAR segmentation and image segmentation using concatenation (`concatenation`) of polygons and spatial join (`sjoin`) of polygons;
2. `assess_results.py`: evaluates results by comparing them with the ground truth, calculates metrics and tags detections. Specify which combination method to assess in the configuration file;
3. `assess_area.py`: calculates the free and occupied surface of each EGID and compares it with the ground truth. Specify which combination method to assess in the configuration file.

### Workflow

The workflow can be run by issuing the following list of commands:

```
python scripts/assessment/combine_results_seg.py config/config_combine_seg.yaml
python scripts/assessment/assess_results.py config/config_combine_seg.yaml
python scripts/assessment/assess_area.py config/config_combine_seg.yaml
```



## Additional developments

The scripts written for additional developments and not conserved in the final workflow can be found in the `sandbox` folder.

### Filtering of the roof parameters

The suitability of a roof to host a solar or vegetated installation can be estimated based on the roof slope and area. The screening of roofs based on those approximated parameters was tested. It was not integrated to this workflow as other teams already work on some more developed version of this screening.

**Data**: This workflow is based on the following layers, available in the [SITG catalog](http://ge.ch/sitg/sitg_catalog/sitg_donnees). <br>
- CAD_BATIMENT_HORSOL_TOIT.shp: Roof areas of above-ground buildings.
- OCEN_SOLAIRE_ID_SURFACE_BASE.shp: roofs, sheds and parkings.
- FTI_PERIMETRE.shp: perimeters of the industrial zones managed by the Foundation for Industrial Lands of Geneva.
- DPS_ENSEMBLE.shp & DPS_CLASSEMENT.shp: architectural and landscape surveys of the canton, archaeological and archival research sites, and scientific inventories. Listed buildings in accordance with the cantonal law on the protection of monuments and sites.

**Requirements**
- There are no hardware or software requirements.
- Everything was tested with Python 3.11.

**Workflow**

<figure align="center">
<image src="img\attribute_filtering_flow.jpeg" alt="Diagram of the methodology" style="width:60%;">
<figcaption align="center">Diagram of the criteria applied to determine the roof suitability for vegetation and solar panels.</figcaption> 
</figure>

All the filters are applied in one script. 

```
python scripts/sandbox/filter_by_attributes.py config/config_expert_attributes.yaml
```

### Collaboration with flai

We worked with [flai](https://www.flai.ai/) to test their algorithm for classifying LiDAR point clouds. flai vectorized the clusters of the class "Roof objects". A script was written to assess the quality of the results based on the vectorized clusters.

```
python scripts/sandbox/assess_flai.py config/config_flai.yaml
```
The path to the config file is hard-coded at the start of each script.
```
python scripts/image_segmentation/optimize_hyperparameters.py config/config_imgseg.yaml
```
