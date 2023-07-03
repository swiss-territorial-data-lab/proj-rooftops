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
INDUSTRIAL_ZONES=INPUT_FILES['industrial_zones']

HOUSE_VS_BUILDING=PARAMETERS['house_vs_building']
FLAT_VS_PITCHED_HOUSES=PARAMETERS['houses']['flat_vs_pitched']
FLAT_SOLAR_HOUSES=PARAMETERS['houses']['flat_roofs']['solar_area']
PITCHED_SOLAR_HOUSES=PARAMETERS['houses']['pitched_roofs']['solar_area']
BUILDING_AREA=PARAMETERS['buildings']['solar_area']
FLAT_VS_PITCHED_INDUSTRY=PARAMETERS['industrial_buildings']['flat_vs_pitched']
FLAT_SOLAR_INDUSTRY=PARAMETERS['industrial_buildings']['flat_roofs']['solar_area']
PITCHED_SOLAR_INDUSTRY=PARAMETERS['industrial_buildings']['pitched_roofs']['solar_area']
VEGETATION_AREA=PARAMETERS['vegetation_area']
VEGETATION_INCLINATION=PARAMETERS['vegetation_inclination']

os.chdir(WORKING_DIRECTORY)
FILEPATH=os.path.join(fct_misc.ensure_dir_exists('processed/roofs'), 'roofs.gpkg')

logger.info('Reading input files...')

roofs=gpd.read_file(ROOFS)
solar_surfaces=gpd.read_file(SOLAR_SURFACES)
heritage_ensemble=gpd.read_file(HERITAGE_ENSEMBLE)
heritage_classement=gpd.read_file(HERITAGE_CLASSEMENT)
industrial_zones=gpd.read_file(INDUSTRIAL_ZONES)

roofs=fct_misc.test_valid_geom(roofs[['OBJECTID', 'geometry']], correct=True, gdf_obj_name='DIT roofs')

nbr_roofs=solar_surfaces.shape[0]
logger.info(f'There are {nbr_roofs} roof shapes for {len(solar_surfaces.EGID.unique().tolist())} EGIDs.')

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

if united_roofs.shape[0]!=nbr_roofs:
    logger.error('The number of roofs changed after the join between DIT and OCEN.' +
                 f' There is a difference of {united_roofs.shape[0]-nbr_roofs} roofs compared to the original number.')

logger.info('Indicating buildings in heritage zones...')

heritage_ensemble.rename(columns={'N_CLASSEME': 'NO_CLASSE'}, inplace=True, errors='raise')
heritage=pd.concat(
    [heritage_classement[['OBJECTID', 'NO_CLASSE', 'geometry']], 
      heritage_ensemble[['OBJECTID', 'NO_CLASSE', 'geometry']]]
)

roofs_with_heritages=gpd.sjoin(united_roofs, heritage, how='left', lsuffix='roof', rsuffix='heritage')
# TODO: determine overlap percentage and set attributes only over a certain limit
# or use predicate = 'within" and then add the attributes for all EGID

roofs_with_heritages['suitability']=None
roofs_with_heritages.loc[roofs_with_heritages['NO_CLASSE'].notna(), 'suitability']='unsure'
roofs_with_heritages['reason']=None
roofs_with_heritages.loc[roofs_with_heritages['NO_CLASSE'].notna(), 'reason']='Part of an heritage building'

roofs_with_heritages.drop_duplicates(subset='OBJECTID_OCEN', ignore_index = True, inplace=True)

if roofs_with_heritages.shape[0]!=nbr_roofs:
    logger.error('The number of roofs changed after the join the heritage geodata.' +
                 f' There is a difference of {roofs_with_heritages.shape[0]-nbr_roofs} roofs compared to the original number.')

logger.info('Marking roofs too steep for vegetation...')
condition_vegetation_inclination=(
     (roofs_with_heritages['PENTE_MOYE'] >= VEGETATION_INCLINATION) & (roofs_with_heritages['suitability'].isnull())
)
roofs_with_heritages.loc[condition_vegetation_inclination, 'suitability'] = 'unsuitable for vegetation'
roofs_with_heritages.loc[condition_vegetation_inclination, 'reason'] = 'slope too steep'

logger.info('Determining which roofs are parts of industrial buildings...')
industrial_zones.drop(columns=['NOM_ZONE', 'SOUS_ZONE', 'SURF_ZONE', 'SHAPE_AREA', 'SHAPE_LEN'], inplace=True)
industrial_zones.rename(columns={'OBJECTID': 'OBJECTID_IZ', 'NOM': 'NOM_ZONE', 'N_ZONE': 'NO_INDUSTRIAL_ZONE'}, inplace=True)

roofs_by_zone=gpd.sjoin(roofs_with_heritages, industrial_zones, 
                        how='left', predicate='within', lsuffix='', rsuffix='industy')
roofs_by_zone.drop_duplicates(subset=roofs_with_heritages.columns, inplace=True, ignore_index=True)

if roofs_by_zone.shape[0]!=nbr_roofs:
    logger.error('The number of roofs changed after the join the industrial zones.' +
                 f' There is a difference of {roofs_by_zone.shape[0]-nbr_roofs} roofs compared to the original number.')
    
logger.info('Sorting the roofs by slope...')
flat_roofs_by_zone_tmp=roofs_by_zone[
    (
        (roofs_by_zone['NO_INDUSTRIAL_ZONE'].isnull()) & (roofs_by_zone['PENTE_MOYE'] < FLAT_VS_PITCHED_HOUSES)
    ) |
    (
        (~roofs_by_zone['NO_INDUSTRIAL_ZONE'].isnull()) & (roofs_by_zone['PENTE_MOYE'] < FLAT_VS_PITCHED_INDUSTRY)
    )
]

pitched_roofs_by_zone_tmp=roofs_by_zone[
    (
        (roofs_by_zone['NO_INDUSTRIAL_ZONE'].isnull()) & (roofs_by_zone['PENTE_MOYE'] >= FLAT_VS_PITCHED_HOUSES)
    ) |
    (
        (~roofs_by_zone['NO_INDUSTRIAL_ZONE'].isnull()) & (roofs_by_zone['PENTE_MOYE'] >= FLAT_VS_PITCHED_INDUSTRY)
    )
]

# Better by area than occurence?
flat_occurences_egid=flat_roofs_by_zone_tmp.value_counts(subset=['EGID'])
pitched_occurences_egid=pitched_roofs_by_zone_tmp.value_counts(subset=['EGID'])
common_egid=flat_occurences_egid.index.intersection(pitched_occurences_egid.index)

logger.info(f'With a limit at {FLAT_VS_PITCHED_HOUSES}° for houses and "skyscraper" and' +
            f' a limit at {FLAT_VS_PITCHED_INDUSTRY}° for industrial buildings,' +
            f' there are {len(flat_occurences_egid)} flat roof buildings and {len(pitched_occurences_egid)} pitched roof buildings. ' +
            f'{len(common_egid)} buildings are in the two classes.')
logger.info(f'Buildings are suppressed in the table where there are the least occurences.')

egid_occurences=pd.concat([flat_occurences_egid, pitched_occurences_egid], axis=1)
egid_occurences[egid_occurences.isna()]=0
egid_occurences.reset_index(inplace=True)

flat_egids=egid_occurences.loc[egid_occurences[0] >= egid_occurences[1], 'EGID'].tolist()
pitched_egids=egid_occurences.loc[egid_occurences[0] < egid_occurences[1], 'EGID'].tolist()
flat_egids.append(0)

flat_roofs_by_zone=flat_roofs_by_zone_tmp[flat_roofs_by_zone_tmp.EGID.isin(flat_egids)]
pitched_roofs_by_zone=pitched_roofs_by_zone_tmp[pitched_roofs_by_zone_tmp.EGID.isin(pitched_egids)]

new_nbr_roofs=flat_roofs_by_zone.shape[0] + pitched_roofs_by_zone.shape[0]
if new_nbr_roofs != nbr_roofs:
    logger.error('The number of roofs changed after the separation between flat and pitched.' +
                 f' There is a difference of {new_nbr_roofs-nbr_roofs} roofs compared to the original number.')

logger.info('Saving file...')
roofs_by_zone.to_file(FILEPATH, layer='attributes_filtered_roofs')

logger.success(f'Done! The results were written in the geopackage {FILEPATH}, in the layer "attributes_filtered_roofs".')

