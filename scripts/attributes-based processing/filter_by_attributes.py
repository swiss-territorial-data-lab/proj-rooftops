import os, sys
from loguru import logger
from yaml import load, FullLoader

import geopandas as gpd
import pandas as pd
from numpy import NaN

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc

logger=fct_misc.format_logger(logger)

logger.info(f"Using config_expert_attributes.yaml as config file.")
with open('config/config_expert_attributes.yaml') as fp:
        cfg = load(fp, Loader=FullLoader)['filter_by_attributes.py']


# Define constants --------------------------

WORKING_DIRECTORY=cfg['working_directory']
INPUT_FILES=cfg['input_files']
PARAMETERS=cfg['parameters']

ROOFS=INPUT_FILES['roofs']
SOLAR_SURFACES=INPUT_FILES['potential_solar_surfaces']
HERITAGE_ENSEMBLE=INPUT_FILES['heritage']['ensembles']
HERITAGE_CLASSEMENT=INPUT_FILES['heritage']['classement']

HOUSE_VS_BUILDING=PARAMETERS['house_vs_building']
FLAT_VS_PITCHED=PARAMETERS['houses']['flat_vs_pitched']
FLAT_SOLAR_AREA=PARAMETERS['houses']['flat_roofs']['solar_area']
PITCHED_SOLAR_AREA=PARAMETERS['houses']['pitched_roofs']['solar_area']
BUILDING_AREA=PARAMETERS['buildings']['solar_area']
VEGETATION_AREA=PARAMETERS['vegetation_area']
VEGETATION_INCLINATION=PARAMETERS['vegetation_inclination']

os.chdir(WORKING_DIRECTORY)
FILEPATH=os.path.join(fct_misc.ensure_dir_exists('processed/roofs'), 'roofs.gpkg')

logger.info('Reading input files...')

roofs=gpd.read_file(ROOFS)
solar_surfaces=gpd.read_file(SOLAR_SURFACES)
heritage_ensemble=gpd.read_file(HERITAGE_ENSEMBLE)
heritage_classement=gpd.read_file(HERITAGE_CLASSEMENT)

roofs=fct_misc.test_valid_geom(roofs[['OBJECTID', 'geometry']], correct=True, gdf_obj_name='DIT roofs')

logger.info('Uniting the roofs as defined by the SITG and the OCEN...')

solar_surfaces['area']=round(solar_surfaces.area, 3)
joined_surfaces=gpd.sjoin(roofs, solar_surfaces, how='right', predicate='intersects', lsuffix='DIT', rsuffix='OCEN')
roofs['geom_DIT']=roofs.geometry
joined_surfaces=joined_surfaces.merge(roofs[['OBJECTID', 'geom_DIT']], how='left', left_on='OBJECTID_DIT', right_on='OBJECTID')

intersecting_area=[]
for (geom1, geom2) in zip(joined_surfaces.geom_DIT.values.tolist(), joined_surfaces.geometry.values.tolist()):
    if geom1 is not None:
        intersecting_area.append(round(geom1.intersection(geom2).area, 3))
    else:
        intersecting_area.append(None)

joined_surfaces['intersecting_area']=intersecting_area
joined_surfaces.drop(columns=['geom_DIT', 'index_DIT'], inplace=True)
joined_surfaces.sort_values(by=['intersecting_area'], ascending=False, inplace=True)

united_roofs=joined_surfaces.drop_duplicates(subset='OBJECTID_OCEN', ignore_index = True)

del roofs, solar_surfaces, joined_surfaces

logger.info('Indicating buildings in heritage zones...')

heritage_ensemble.rename(columns={'N_CLASSEME': 'NO_CLASSE'}, inplace=True, errors='raise')
heritage=pd.concat(
    [heritage_classement[['OBJECTID', 'NO_CLASSE', 'geometry']], 
      heritage_ensemble[['OBJECTID', 'NO_CLASSE', 'geometry']]]
)

roofs_with_heritages=gpd.sjoin(united_roofs, heritage, how='left', lsuffix='roof', rsuffix='heritage')
# TODO: determine overlap percentage and set attributes only over a certain limit

roofs_with_heritages['suitability']=None
roofs_with_heritages.loc[roofs_with_heritages['NO_CLASSE'].notna(), 'suitability']='unsure'
roofs_with_heritages['reason']=None
roofs_with_heritages.loc[roofs_with_heritages['NO_CLASSE'].notna(), 'reason']='Part of an heritage building'

roofs_with_heritages.drop_duplicates(subset='OBJECTID_OCEN', ignore_index = True)

roofs_with_heritages.to_file(FILEPATH, layer='attributes_filtered_roofs')

logger.success(f'Done! The results were written in the geopackage {FILEPATH}, in the layer "attributes_filtered_roofs".')

