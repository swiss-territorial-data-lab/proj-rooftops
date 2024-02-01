# Delimitation of the free areas on rooftops

Goal: Determine the space available on rooftops by detecting objects. Production of a binary (free/occupied) vector layer.

**Table of content**

- [Requirements](#requirements)
	- ...

## Requirements

### Hardware

For the processing of the images, a CUDA-compatible GPU is needed.
For the processing of the LiDAR point cloud, there is no hardware requirement.

### Installation

The scripts have been developed with Python 3.8 using PyTorch version 1.10 and CUDA version 11.3.

All the dependencies required for the project are listed in `requirements.in` and `requirements.txt`. To install them:

- Create a Python virtual environment

        $ python3 -m venv <dir_path>/<name of the virtual environment>
        $ source <dir_path>/<name of the virtual environment>/bin/activate

- Install dependencies

        $ pip install -r requirements/requirements.txt

    - Requirements for the image workflow only

            $ pip install -r requirements/requirements_LiDAR.txt

    - Requirements for the LiDAR workflow only

            $ pip install -r requirements/requirements_images.txt


_requirements.txt_ has been obtained by compiling _requirements.in_. Recompiling the file might lead to libraries version changes:

        $ pip-compile requirements.in

## Image segmentation


## Classification of occupation

**Goal**: Classify the roof planes as occupied or potentially free based on their roughness and intensity.

**Data**: LiDAR point cloud with intensity values. Here, the [the 2019 flight over the Geneva canton](https://ge.ch/sitggeoportal1/apps/webappviewer/index.html?id=311e4a8ae2724f9698c9bcfb6ab45c56) was used.

**Workflow**

(*facultative*) The script `get_lidar_infos.py` allows to get some characteristics of the point clouds.

Run the following command lines to perform the LiDAR processing:

```
python scripts/lidar_products/rasterize_intensity.py config/config_lidar_products.yaml
python scripts/lidar_products/rasterize_roughness.py config/config_lidar_products.yaml
python scripts/lidar_products/get_zonal_stats.py config/config_lidar_products.yaml
````

The command lines perform the following steps:
1. Create an intensity raster for each LiDAR point cloud in the input directory.
    - The parameters and the function for the raster of intensity are referenced here: [LidarIdwInterpolation - WhiteboxTools](https://www.whiteboxgeo.com/manual/wbt_book/available_tools/lidar_tools.html#LidarIdwInterpolation)
2. Create a DEM and save it in a raster. Then estimate the multi-scale roughness from the DEM.
    - The parameters and the function for the DEM are referenced here: [LidarDigitalSurfaceModel - WhiteboxTools](https://www.whiteboxgeo.com/manual/wbt_book/available_tools/lidar_tools.html#LidarDigitalSurfaceModel)
    - The parameters and the function for the multi-scale roughness are referenced here: [MultiscaleRoughness - WhiteboxTools](https://www.whiteboxgeo.com/manual/wbt_book/available_tools/geomorphometric_analysis.html#MultiscaleRoughness)
3. Get zonal stats of intensity and roughness for roof planes.
    - Only roof planes larger than 2 m<sup>2</sup> are classified as occupied, because they a.

When *no ground truth is available*, the classification can be performed with the script `filter_surfaces_by_attributes.py` using thresholds calibrated manually by an operator. The results can then eventually be assessed by experts, their quality assessed, and used as ground truth.

```
python scripts/lidar_products/filter_surfaces_by_attributes.py config/config_lidar_products.yaml
python scripts/assessment/assess_classif_surfaces.py config/config_lidar_products.yaml
```

When *a ground truth is available*, the classification can be performed and assessed with the script `random_forest.py`.

```
python scripts/lidar_products/random_forest.py config/config_lidar_products.yaml
```

The other scripts are some attempts to detect objects based on intensity. The results were not as good as expected and were therefore not implemented.


## LiDAR segmentation


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
- The necessary libraries can be installed from the file `requirements.txt`.

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
