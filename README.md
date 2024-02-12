# Delimitation of the free areas on rooftops

Goal: Determine the space available on rooftops by detecting objects. Production of a binary (free/occupied) vector layer.

**Table of content**

- [Requirements](#requirements)
	- [Hardware](#hardware)
    - [Installation](#installation)
- [Floder structure](#folder-structure)
- [Classification of occupancy](#classification-of-the-roof-plane-occupation)
- [LiDAR segmentation](#lidar-segmentation)
- [Image segmentation](#image-segmentation)

## Requirements

### Hardware

For the processing of the *images*, a CUDA-compatible GPU is needed. <br>
For the processing of the *LiDAR point cloud*, there is no hardware requirement.

The scripts dedicated to image segmentation were run on a 32 GiB RAM machine with 16 GiB GPU (NVIDIA Tesla T4)

### Installation

The scripts were developed with Python 3.8<!-- 3.10 actually for the pcdseg --> on Unbuntu 20.04 OS. 

- For image processing, the following specific libraries were used:
    - PyTorch version 1.10
    - CUDA version 11.3
    - segment-geospatial [0.10.2](https://github.com/opengeos/segment-geospatial/releases/tag/v0.10.2)

All the dependencies required for the project are listed in `requirements.in` and `requirements.txt`. To install them:

- Create a Python virtual environment

```
$ python3 -m venv <dir_path>/<name of the virtual environment>
$ source <dir_path>/<name of the virtual environment>/bin/activate
```

- Install dependencies

```
$ pip install -r requirements.txt
```

_requirements.txt_ has been obtained by compiling _requirements.in_. Recompiling the file might lead to libraries version changes:

```
$ pip-compile requirements.in
```

- The library `segment-geospatial` was adapted to our needs. The modified version can be clone from this forked repository: https://github.com/swiss-territorial-data-lab/segment-geospatial.git. To install it in your virtual environment execute the following commands:

```
$ cd segment-geospatial
$ git checkout ch/dev
$ pip install .
```

or in editable mode:

```
$ pip install -e .
```

If the installation is successful, the message "You are using a modified version of segment-geospatial library (v 0.10.2 fork)" must be printed in the prompt while executing the script `segment_images.py`.  

## Folder structure

<pre>.
├── config                                         
│   ├── config_combine_seg.yaml                 # combine img and pcd segmentation results
│   ├── config_imgseg.yaml                      # img segmentation workflow
│   ├── config_pcdseg_all_roofs.yaml            # pcd workflow (all roofs)
│   └── config_pcdseg_pitched_roofs.yaml        # pcd workflow (pitched roofs)
├── scripts 
│   ├── assessment    
│   │   ├── assess_area.py                      # compute occupied and free area (detections + GT)
│   │   ├── assess_results.py                   # compute metrics
│   │   ├── combine_results_lidar.py            # combine lidar classification and pcd segmentation vectors
│   │   └── combine_results_seg.py              # combine img and pcd segmentation vectors
│   ├── functions                         
│   │   ├── fct_figures.py                          
│   │   ├── fct_metrics.py   
│   │   ├── fct_misc.py   
│   │   ├── fct_optimization.py   
│   │   └── fct_pcdseg.py   
│   ├── image_segmentation               
│   │   ├── filter_merge_detections.py           # filter and merge polygons obtained from segment_images.py  
│   │   ├── generate_tiles.py                    # produce image tiles per building
│   │   ├── optimize_hyperparameters.py          # optimize SAM hyperparameters with optuna
│   │   └── segment_images.py                    # segment objects in images with SAM + mask vectorization
│   ├── pcd_segmentation              
│   │   ├── optimize_hyperparam_LIDAR.py    
│   │   ├── pcd_segmentation.py   
│   │   ├── post_processing.py   
│   │   ├── prepare_data.py  
│   │   └── vectorization.py   
│   └── sandbox      
│       ├── assess_synthetic_examples.py         # development of "charges" detection count with synthetic vectors
│       ├── fuse_las_files.py   
│       └── mask_for_buildings.py                # mask all image elements except roofs
├── .gitignore                                      
├── LICENSE
├── README.md                                      
├── requirements.in                             # list of python libraries required for the project                           
└── requirements.txt                            # python dependencies compiled from requirements.in file                            
</pre>

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
1. `prepare_data.py`: read and filter the 3D point cloud data to keep the roofs of the selected EGIDs,
2. `pcd_segmentation.py`: segment in planes and clusters the point cloud data,
3. `vectorization.py`: create 2D polygons from the segmented point cloud data,
7. `post_processing.py`: merge the results for the pitched and general roofs together and simplify the geometry of the detections.
5. `assess_results.py`: Evaluate the results based on the ground truth,
6. `assess_area.py`: Calculate the free and occupied surface of each EGIDs and compare it with the ground truth.

The workflow described here is working with the training subset of the ground truth. The configuration file `config-pcdseg_test.yaml` works with the test subset of the ground truth.

## Image segmentation

### Overview
The set of scripts is dedicated to segmenatation of objects in images. Tiles fitting to the extent of the selected buildings (by EGID) are produced. Images are segmented using [segment-geospatial](https://github.com/opengeos/segment-geospatial) which provides a practical framework to using [SAM](https://github.com/facebookresearch/segment-anything) (Segment-Anything Model) with georeferenced data. Detection masks are converted to vectors, filtered and merged to produce a vector layer of detected objects on the roofs of selected buildings. Finally, the results are evaluated by comparing them with ground truth labels. 

### Data

The image segmentation workflow uses the following input data:

- True orthophotos of the canton of Geneva: processed from aiborne acquisitions by the State of Geneva in 2019. RGB tiff images with a spatial resolution of about 7 cm/px. Images are available on request from SITG. Saved locally for the STDL in /mnt/s3/proj-rooftops/02_Data/initial/Geneva/ORTHOPHOTOS/2019/TIFF_TRUEORTHO/*.tiff
- Image tile shapes: vector shapefile of the true orthophoto tiles available on request from SITG. Saved locally for the STDL in /mnt/s3/proj-rooftops/02_Data/initial/Geneva/ORTHOPHOTOS/2019/TUILES_TRUEORTHO/Tuiles.shp
- Roof delimitation: vector shapefile [CAD_BATIMENT_HORSOL_TOIT.shp](https://ge.ch/sitg/sitg_catalog/sitg_donnees?keyword=&geodataid=0635&topic=tous&service=tous&datatype=tous&distribution=tous&sort=auto) providing roof planes by EGID. 
- Ground truth objects: vector shapefile of the ground truth labels: /mnt/s3/proj-rooftops/02_Data/ground_truth/occupation/PanData/roofs_STDL_proofed_2023-11-13.shp
- EGIDs lists: list of buildings (EGID) selected for this project and saved in /mnt/s3/proj-rooftops/02_Data/ground_truth/PanData/occupation/Partition/. The buildings are split into different datasets:
    - EGIDs_GT_full.csv: list of 122 EGIDs selected to established the labels vectorization
    - EGIDs_GT_test.csv: list of 17 EGIDs selected to control the performance of the algorithm on a test dataset.
    - EGIDs_GT_training.csv: list of 105 EGIDs selected to perform hyperparameter optimization of algorithms on a training dataset. 
    - EGIDs_GT_training_subsample_imgseg.csv: In the case of image segmentation, the training list is too long to perform hyperparameters optimization in a reasonable time. For this reason, a training list reduced to 25 buildings is proposed. 

### Script description

1. `generate_tiles.py`: produces customized tiles of the roof extent
2. `segment_images.py`: produces detection masks and vectorizes them
3. `produce_vector_layer.py`: filters the vector layer for each building and aggregates all layers into a single one (detected objects)
4. `assess_results.py`: evaluates results by comparing them with the ground truth, computes metrics and tags detections
5. `optimize_hyperparameters.py`: optimizes SAM hyperparameters to maximize the desired metrics (f1-score, median IoU, precision,...). Based on [Oputna](https://optuna.org/) framework

### Workflow

The workflow can be run by issuing the following list of commands:

```
$ python scripts/image_segmentation/generate_tiles.py config/config_imgseg.yaml
$ python scripts/image_segmentation/segment_images.py config/config_imgseg.yaml
$ python scripts/image_segmentation/filter_merge_detections.py config/config_imgseg.yaml
$ python scripts/assessment/assess_results.py config/config_imgseg.yaml
$ python scripts/assessment/assess_area.py config/config_imgseg.yaml
```

The optimization of hyperparameters can be performed as follow:

```
$ python scripts/image_segmentation/optimize_hyperparameters.py config/config_imgseg.yaml
```