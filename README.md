# proj-rooftops


**Data**: This workflow is based on some classified LiDAR point cloud.
**Requirements**:
- There are no hardware or software requirements for this. Everything was tested with Python 3.11
- The necessary libraries can be installed from the file `requirements.txt`

The path to the config file is hard-coded at the start of each script.

## Workflow
(*facultative*) The script `get_lidar_infos.py` allows to get some characteristics of the point clouds.

```
python rasterize_intensity.py
python rasterize_roughness.py
python filter_surfaces_by_attributes.py
```

The command lines perform the following steps:
1. Create an intensity raster for each LiDAR point cloud in the input directory.
    - The parameters and the function for the raster of intensity are referenced here: [LidarIdwInterpolation - WhiteboxTools](https://www.whiteboxgeo.com/manual/wbt_book/available_tools/lidar_tools.html#LidarIdwInterpolation)
2. Create a DEM and save it in a raster. Then estimate the multi-scale roughness from the DEM.
    - The parameters and the function for the DEM are referenced here: [LidarDigitalSurfaceModel - WhiteboxTools](https://www.whiteboxgeo.com/manual/wbt_book/available_tools/lidar_tools.html#LidarDigitalSurfaceModel)
    - The parameters and the function for the multi-scale roughness are referenced here: [MultiscaleRoughness - WhiteboxTools](https://www.whiteboxgeo.com/manual/wbt_book/available_tools/geomorphometric_analysis.html#MultiscaleRoughness)
3. Classify the roof section by estimating their degree of occupation according to their roughness and intensity.

The other scripts are attempts to detect objects based on intensity. The results were not as good as expected.