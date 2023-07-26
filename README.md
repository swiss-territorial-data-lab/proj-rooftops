# proj-rooftops

## Overview

Scripts allowing to:
1. `prepare_data.py`: read and filter the 3D point cloud data
2. `pcd_segmentation.py`: segment in planes and clusters the point cloud data
3. `vectorization.py`: create 2D polygons from the segmented point cloud data
4. `assess_results.py`: Evaluate the results

## Data

Input lidar data are located here: /mnt/s3/proj-rooftops/02_Data/initial/Geneva/LiDAR/2019/
Input roof shapefile is locadted here: /mnt/s3/proj-rooftops/02_Data/initial/Geneva/SHP_CAD_BATIMENT_HORSOL_TOIT
Input GT is located here: /mnt/s3/proj-rooftops/02_Data/ground_truth/

## Comments

For now, the workflow works for a single building. EGID number of the building must be provided + the associated lidar tiles name. The GT for the given building must also be provided.

## Workflow instructions

Following the end-to-end, the workflow can be run by issuing the following list of actions and commands:

Clone `proj-rooftops` repository.

    $ cd proj-rooftops/
    $ python3 -m venv <dir_path>/[name of the virtual environment]
    $ source <dir_path>/[name of the virtual environment]/bin/activate
    $ pip install -r requirements.txt

Adapt the paths and input values of the configuration files accordingly.

    $ python3 scripts/pcd_segmentation/prepare_data.py config/config-pcdseg.yaml
    $ python3 scripts/pcd_segmentation/pcd_segmentation.py config/config-pcdseg.yaml
    $ python3 scripts/pcd_segmentation/vectorization.py config/config-pcdseg.yaml
    $ python3 scripts/pcd_segmentation/assess_results.py config/config-pcdseg.yaml

## Improvements

- The workflow works for a single building, it should be generalized to a list a building
- Either the workflow is processing EGID by EGID either the workflow could process several EGID at the same time. There is LiDAR segmentation tools in whitebox-tools that we can look at as example. I did a test with "lidar_segmentation" (https://www.whiteboxgeo.com/manual/wbt_book/available_tools/lidar_tools.html#LidarSegmentation) on QGIS and it was working well.
- If we process EGID by EGID, the lidar tiles must be automatically associated to the building EGID. Merging of lidar data will be necessary for building spanning over several tiles
- The hyper parameters value must be optimized for several buildings
- The assessment script can be improved 
- Compressed LiDAR data can be stored on S3 and the script can fetch directly from there. 