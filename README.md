# Delimitation of the free areas on rooftops

Goal: Determine the space available on rooftops by detecting objects. Production of a binary (free/occupied) vector layer.

**Table of content**

- [Requirements](#requirements)
	- ...

## Requirements

### Hardware

The scripts have been run with Ubuntu 20.04 OS on a 32 GiB RAM machine with 16 GiB GPU (NVIDIA Tesla T4)

### Installation

The scripts have been developed with Python 3.8 using PyTorch version 1.10 and CUDA version 11.3.

All the dependencies required for the project are listed in `requirements.in` and `requirements.txt`. To install them:

- Create a Python virtual environment

        $ python3 -m venv <dir_path>/[name of the virtual environment]
        $ source <dir_path>/[name of the virtual environment]/bin/activate

- Install dependencies

        $ pip install -r requirements/requirements.txt

    - Requirements for the image workflow only

            $ pip install -r requirements/requirements_LiDAR.txt

    - Requirements for the LiDAR workflow only

            $ pip install -r requirements/requirements_images.txt


_requirements.txt_ has been obtained by compiling _requirements.in_. Recompiling the file might lead to libraries version changes:

        $ pip-compile requirements.in

## Image segmentation


## LiDAR classification


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

```
python scripts\attributes-based_processing\filter_by_attributes.py config/config_expert_attributes.yaml
```

**Workflow**

<figure align="center">
<image src="img\attribute_filtering_flow.jpeg" alt="Diagram of the methodology" style="width:60%;">
<figcaption align="center">Diagram of the criteria applied to determine the roof suitability for vegetation and solar panels.</figcaption> 
</figure>

All the filters are applied in one script. 

```
python filter_by_attributes.py
```

### Collaboration with flai

We worked with [flai](https://www.flai.ai/) to test their algorithm for classifying LiDAR point clouds. flai vectorized the clusters of the class "Roof objects". A script was written to assess the quality of the results based on the vectorized clusters.

```
python scripts/sandbox/assess_flai config/config_flai.yaml
```
