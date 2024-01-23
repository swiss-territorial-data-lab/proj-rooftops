# Delimitation of the free areas on rooftops

Goal: Determine the space available on rooftops by detecting objects. Production of a binary (free/occupied) vector layer.

**Table of content**

- [Requirements](#requirements)
	- [Hardware](#hardware)
    - [Installation](#installation)
- [Image segmentation](#image-segmentation)
- [LiDAR-based classification](#lidar-based-classification)
- [LiDAR segmentation](#lidar-segmentation)
    - [Data](#data)
    - [Workflow](#workflow)

## Requirements

### Hardware

For the processing of the *images*, a CUDA-compatible GPU is needed. <br>
For the processing of the *LiDAR point cloud*, there is no hardware requirement.

### Installation

The scripts have been developed with Python 3.8<!-- 3.10 actually for the pcdseg -->. For the image processing, PyTorch version 1.10 and CUDA version 11.3 were used.

All the dependencies required for the project are listed in `requirements.in` and `requirements.txt`. To install them:

- Create a Python virtual environment

        $ python3 -m venv <dir_path>/[name of the virtual environment]
        $ source <dir_path>/[name of the virtual environment]/bin/activate

- Install dependencies

        $ pip install -r requirements/requirements.txt

    - Requirements for the image workflow only

            $ pip install -r requirements/requirements_images.txt

    - Requirements for the LiDAR workflow only

            $ pip install -r requirements/requirements_LiDAR.txt


_requirements.txt_ has been obtained by compiling _requirements.in_. Recompiling the file might lead to libraries version changes:

        $ pip-compile requirements.in

## Image segmentation

## LiDAR-based classification

## LiDAR segmentation

### Data

- LiDAR point cloud. Here, the [tiles of the 2019 flight over the Geneva canton](https://ge.ch/sitggeoportal1/apps/webappviewer/index.html?id=311e4a8ae2724f9698c9bcfb6ab45c56) were used.
- Delimitation of the roof for each EGID.

This workflow was tested with a ground truth produced for this project. The ground truth is available ... <br>
The ground truth is split into the training and test set to see if the algorithm performs equally on new buildings.

### Workflow

First, the segmentation is performed with different parameters on buildings with a pitched roof than on other buildings.

```
python scripts/pcd_segmentation/prepare_data.py config\config_pcdseg_all_roofs.yaml
python scripts/pcd_segmentation/pcd_segmentation.config\config_pcdseg_all_roofs.yaml
python scripts/pcd_segmentation/vectorization.py config\config_pcdseg_all_roofs.yaml
python scripts/pcd_segmentation/prepare_data.py config\config_pcdseg_pitched_roofs.yaml
python scripts/pcd_segmentation/pcd_segmentation.config\config_pcdseg_pitched_roofs.yaml
python scripts/pcd_segmentation/vectorization.py config\config_pcdseg_pitched_roofs.yaml
```

Then, the results for the pitched roofs and the general results are merged. Their geometry is also simplified with a buffer and copping operation, as well as with the Visvalingam-Wyatt algorithm. The different obstacles are merged together to form the occupied surfaces.

```
python scripts\pcd_segmentation\post_processings.py config\config_pcdseg_all_roofs.yaml
```

Finally, the results are assessed

```
python scripts/pcd_segmentation/assess_results.py config/config-pcdseg.yaml
python scripts/assessment/assess_area.py config/config-pcdseg.yaml
```

More in details, the scripts used above perform the following steps:
1. `prepare_data.py`: read and filter the 3D point cloud data to keep the roofs of the selected EGIDs,
2. `pcd_segmentation.py`: segment in planes and clusters the point cloud data,
3. `vectorization.py`: create 2D polygons from the segmented point cloud data,
5. `assess_results.py`: Evaluate the results based on the ground truth,
6. `assess_area.py`: Calculate the free and occupied surface of each EGIDs and compare it with the ground truth.

The workflow described here is working with the training subset of the ground truth. The configuration file `config-pcdseg_test.yaml` works with the test subset of the ground truth.

