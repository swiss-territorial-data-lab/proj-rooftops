import os
import sys
from glob import glob

import whitebox
# whitebox.download_wbt(linux_musl=True, reset=True)        # Uncomment if issue with GLIBC library
wbt = whitebox.WhiteboxTools()

WORKING_DIR = '/home/gsalamin/Documents/project_data/rooftops/processed/las_files/test_set'


os.chdir(WORKING_DIR)

las_files = glob('*.las')

las_list = ''
for las in las_files:
    las_list = las_list + ', ' + os.path.join(WORKING_DIR, las)
las_list=las_list.lstrip(', ')

outpath = os.path.join(WORKING_DIR, 'test_buildings.las')

wbt.lidar_join(
    las_list, 
    outpath,
)