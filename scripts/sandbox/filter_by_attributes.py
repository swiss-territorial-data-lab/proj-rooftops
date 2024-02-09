import os
import sys
from argparse import ArgumentParser
from loguru import logger
from yaml import load, FullLoader

import geopandas as gpd
import pandas as pd
from numpy import NaN

sys.path.insert(1, 'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)

# Argument and parameter specification
parser = ArgumentParser()
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")
with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]


# Define functions -------------------------

def check_solar_suitability(gdf_by_egid_and_zone):
    """Do the if-else statement to test if there is enough space for solar panels.

    Args:
        gdf_by_egid_and_zone (geodataframe): roofs with the area over the EGID, the type of object, the type of roof, the suitability and the reason.

    Returns:
        geodataframe: roofs with the suitability and reason updated for solar.
    """

    solar_suitability = []
    solar_reason = []

    for roof in gdf_by_egid_and_zone.itertuples():

        if roof.EGID_type == 'flat':
            limits = {'min_area': min([FLAT_SOLAR_HOUSES, FLAT_SOLAR_INDUSTRY]), 'solar_house': FLAT_SOLAR_HOUSES, 'solar_industry': FLAT_SOLAR_INDUSTRY}
        else:
            limits = {'min_area': min([PITCHED_SOLAR_HOUSES, PITCHED_SOLAR_INDUSTRY]), 'solar_house': PITCHED_SOLAR_HOUSES, 'solar_industry': PITCHED_SOLAR_INDUSTRY}

        # The building does not have any other contraindications.
        if roof.suitability == SUITABILITY_MESSAGES['ok']:
            # The roof has at least the min area.
            if roof.available_area_over_EGID < limits['min_area']:
                solar_suitability.append(SUITABILITY_MESSAGES['no solar'])
                solar_reason.append(f'The roof area over the EGID is less than {limits["min_area"]} m2 which is too small for solar panels.')
                continue

            # The building is NOT industrial and too small for solar panels.
            if (roof.object_type != 'industrial_space') & (roof.available_area_over_EGID < limits['solar_house']):
                solar_suitability.append(SUITABILITY_MESSAGES['no solar'])
                solar_reason.append(f'The roof area over the EGID is less than {limits["solar_house"]} m2 which is too small for solar panels.')
                continue
            
            # The building is industrial. and too small for solar panels.
            elif (roof.object_type == 'industrial_space') & (roof.available_area_over_EGID < limits['solar_industry']):
                    solar_suitability.append(SUITABILITY_MESSAGES['no solar'])
                    solar_reason.append(f'The roof area over the EGID is less than {limits["solar_industry"]} m2 which is too small for solar panels.')
                    continue

        # The building is already unsuited for vegetation.
        elif roof.suitability == SUITABILITY_MESSAGES['no vegetation']:

            # The roof has at least the min area
            if roof.available_area_over_EGID < limits['min_area']:
                solar_suitability.append(SUITABILITY_MESSAGES['nothing'])
                solar_reason.append(roof.reason + 
                                    f' The roof area of the EGID is less than {limits["min_area"]} m2, which is too small for solar panels.')
                continue

            # The building is NOT industrial and too small for solar panels.
            if (roof.object_type != 'industrial_space') & (roof.available_area_over_EGID < limits['solar_house']):
                solar_suitability.append(SUITABILITY_MESSAGES['nothing'])
                solar_reason.append(roof.reason + 
                                    f' The roof area of the EGID is less than {limits["solar_house"]} m2, which is too small for solar panels.')
                continue
            
            # The building is industrial.
            elif (roof.object_type == 'industrial_space') & (roof.available_area_over_EGID < limits['solar_industry']):
                    solar_suitability.append(SUITABILITY_MESSAGES['nothing'])
                    solar_reason.append(roof.reason + 
                                        f' The roof area is less than {limits["solar_industry"]} m2, which is too small for solar panels.')
                    continue


        solar_suitability.append(roof.suitability)
        solar_reason.append(roof.reason)


    gdf_by_egid_and_zone['suitability'] = solar_suitability
    gdf_by_egid_and_zone['reason'] = solar_reason

    return gdf_by_egid_and_zone

def set_heritage_status(roofs_gdf, condition):
    """Set the heritage status and information to the roofs of the geodataframe filling the condition.

    Args:
        roofs_gdf (GeoDataFrame): GeoDataFrame with the columns 'object_type', 'suitability' and 'reason'.
        condition (boolean Serie): condition to apply to the GeoDataFrame to determine if the elements are part of an heritage or not.
    """
     
     # Set type
    roofs_gdf.loc[heritage_condition, 'object_type'] = 'heritage space'
    # Set suitability
    roofs_gdf.loc[heritage_condition & (roofs_gdf.suitability != SUITABILITY_MESSAGES['nothing']), 'suitability'] = SUITABILITY_MESSAGES['uncertain']

    # Set reason
    condition = (roofs_gdf.suitability == SUITABILITY_MESSAGES['uncertain']) & (roofs_gdf.reason.isnull())
    roofs_gdf.loc[condition, 'reason'] = 'This EGID is part of an heritage building'
    condition = (roofs_gdf.suitability == SUITABILITY_MESSAGES['uncertain']) & (~roofs_gdf.reason.isnull())
    roofs_gdf.loc[condition, 'reason'] = roofs_gdf.loc[condition, 'reason'] + ' This EGID is part of an heritage building.'

    return roofs_gdf


# Define constants --------------------------

WORKING_DIRECTORY = cfg['working_directory']
OUTPUT_DIRECTORY = cfg['output_directory']
INPUT_FILES = cfg['input_files']
PARAMETERS = cfg['parameters']

ROOFS = INPUT_FILES['roofs']
SOLAR_SURFACES = INPUT_FILES['potential_solar_surfaces']
HERITAGE_ENSEMBLE = INPUT_FILES['heritage']['ensembles']
HERITAGE_CLASSEMENT = INPUT_FILES['heritage']['classement']
INDUSTRIAL_ZONES = INPUT_FILES['industrial_zones']

SOLAR_ABS_MIN_AREA = PARAMETERS['solar_absolute_minimum_area']
HOUSE_VS_BUILDING = PARAMETERS['house_vs_building']
FLAT_VS_PITCHED_HOUSES = PARAMETERS['houses']['flat_vs_pitched']
FLAT_SOLAR_HOUSES = PARAMETERS['houses']['flat_roofs']['solar_area']
PITCHED_SOLAR_HOUSES = PARAMETERS['houses']['pitched_roofs']['solar_area']
BUILDING_AREA = PARAMETERS['buildings']['solar_area']
FLAT_VS_PITCHED_INDUSTRY = PARAMETERS['industrial_buildings']['flat_vs_pitched']
FLAT_SOLAR_INDUSTRY = PARAMETERS['industrial_buildings']['flat_roofs']['solar_area']
PITCHED_SOLAR_INDUSTRY = PARAMETERS['industrial_buildings']['pitched_roofs']['solar_area']
VEGETATION_AREA = PARAMETERS['vegetation_area']
VEGETATION_INCLINATION = PARAMETERS['vegetation_inclination']

SUITABILITY_MESSAGES = {'ok': 'suitable for vegetation and solar panels', 'no vegetation': 'unsuitable for vegetation', 'no solar': 'unsuitable for solar panels', 
                      'nothing': 'not suitable for valorization', 'uncertain': 'unsure'}

os.chdir(WORKING_DIRECTORY)
FILEPATH = os.path.join(misc.ensure_dir_exists(OUTPUT_DIRECTORY), 'classified_roofs.gpkg')

logger.info('Read input files...')

roofs = gpd.read_file(ROOFS)
solar_surfaces = gpd.read_file(SOLAR_SURFACES)
heritage_ensemble = gpd.read_file(HERITAGE_ENSEMBLE)
heritage_classement = gpd.read_file(HERITAGE_CLASSEMENT)
industrial_zones = gpd.read_file(INDUSTRIAL_ZONES)

roofs = misc.check_validity(roofs[['OBJECTID', 'geometry', 'EGID']], correct=True)
solar_surfaces = misc.check_validity(
    solar_surfaces[['OBJECTID', 'EGID', 'TYPE_SURFA', 'ID_SURFACE', 'ORIENTATIO', 'PENTE_MOYE', 'IRR_MOYENN', 'SURFACE_TO', 'geometry']], 
    correct=True
)

logger.info('Merge the roofs as defined by the DIT and the OCEN...')

solar_surfaces['area'] = round(solar_surfaces.area, 3)
joined_surfaces = gpd.sjoin(
    roofs[['OBJECTID', 'geometry']], solar_surfaces, how='right', predicate='intersects', lsuffix='DIT', rsuffix='OCEN'
)
roofs['geom_DIT'] = roofs.geometry
joined_surfaces_with_area = joined_surfaces.merge(roofs[['OBJECTID', 'geom_DIT']], how='left', left_on='OBJECTID_DIT', right_on='OBJECTID')

# Determine the percentage of each OCEN roof area covered by each intersecting DIT roof.
intersecting_area = []
for (geom1, geom2) in zip(joined_surfaces_with_area.geom_DIT.values.tolist(), joined_surfaces_with_area.geometry.values.tolist()):
    if geom1 is not None:
        intersecting_area.append(round(geom1.intersection(geom2).area/geom2.area, 3))
    else:
        intersecting_area.append(None)

joined_surfaces_with_area['intersecting_area'] = round(
    joined_surfaces_with_area.geom_DIT.intersection(joined_surfaces_with_area.geometry).area/joined_surfaces_with_area.geometry.area, 3
)
joined_surfaces_with_area.sort_values(by=['intersecting_area'], ascending=False, inplace=True)

united_surfaces = joined_surfaces_with_area.drop_duplicates(subset='OBJECTID_OCEN', ignore_index=True)

# Determine valid intersections (subjective threshold)
captured_dit_id = united_surfaces['OBJECTID_DIT'].unique().tolist() + \
    joined_surfaces_with_area.loc[
        joined_surfaces_with_area.geometry.intersection(joined_surfaces_with_area.geom_DIT).area / joined_surfaces_with_area.geom_DIT.area >= 0.50, 'OBJECTID_DIT'
    ].unique().tolist()
missed_DIT_roofs = roofs[~roofs['OBJECTID'].isin(captured_dit_id)].copy()

united_surfaces.drop(columns=['geom_DIT', 'index_DIT', 'OBJECTID'], inplace=True)
united_surfaces.loc[united_surfaces['intersecting_area'] < 0.75, 'OBJECTID_DIT'] = NaN

# Tag DIT roofs with no OCEN correspondance.
missed_DIT_roofs.rename(columns={'OBJECTID': 'OBJECTID_DIT'}, inplace=True)
missed_DIT_roofs.drop(columns=['geom_DIT'], inplace=True)
missed_DIT_roofs['suitability'] = 'unknown'
missed_DIT_roofs['reason'] = 'This roof has no correspondence with the OCEN roofs.'
missed_DIT_roofs.loc[missed_DIT_roofs.area < SOLAR_ABS_MIN_AREA, 'suitability'] = SUITABILITY_MESSAGES['nothing']
missed_DIT_roofs.loc[missed_DIT_roofs.area < SOLAR_ABS_MIN_AREA, 'reason'] = f'The roof section is less than {SOLAR_ABS_MIN_AREA} m2, which is too small for a solar panel.'
logger.info(f'{missed_DIT_roofs.shape[0]} DIT roofs do not correspond to the OCEN roofs.')

nbr_surfaces = united_surfaces.shape[0]
logger.info(f'There are {nbr_surfaces} roof shapes for {len(united_surfaces.EGID.unique().tolist())} different EGIDs.')

del roofs, solar_surfaces, joined_surfaces


logger.info('Set a building type...')
united_surfaces['object_type'] = united_surfaces.TYPE_SURFA.copy()
united_surfaces.loc[(united_surfaces.object_type == 'toiture') & (united_surfaces.EGID == 0), 'object_type'] = 'toiture sans EGID'

united_surfaces['suitability'] = SUITABILITY_MESSAGES['ok']
united_surfaces['reason'] = None


logger.info('Set suitability for vegetation')
logger.info('    based on roof slope...')

condition = united_surfaces.PENTE_MOYE > VEGETATION_INCLINATION
united_surfaces.loc[condition, 'suitability'] = SUITABILITY_MESSAGES['no vegetation']
united_surfaces.loc[condition, 'reason'] = 'The slope is too steep for vegetation.'

logger.info('    based on EGID area...')

# Sum area available for vegetation over EGID
available_surfaces_EGID = united_surfaces[(united_surfaces.EGID != 0) & (united_surfaces.suitability == SUITABILITY_MESSAGES['ok'])].copy()
area_tmp_gdf = available_surfaces_EGID.groupby(by='EGID')['SURFACE_TO'].sum().reset_index()
area_tmp_gdf.rename(columns={'SURFACE_TO': 'available_area_over_EGID'}, inplace=True)

# Take back the surfaces with EGID = 0 that are unsuitable for vegetation
united_surfaces_with_area = united_surfaces.merge(area_tmp_gdf, on='EGID', how='left', suffixes=('_over_EGID', ''))
condition = united_surfaces_with_area.available_area_over_EGID.isnull() & (united_surfaces_with_area.suitability == SUITABILITY_MESSAGES['ok'])
united_surfaces_with_area.loc[condition, 'available_area_over_EGID'] = united_surfaces_with_area.loc[condition, 'SURFACE_TO']

condition = (united_surfaces_with_area.suitability == SUITABILITY_MESSAGES['ok']) & (united_surfaces_with_area.available_area_over_EGID < VEGETATION_AREA)
united_surfaces_with_area.loc[condition, 'suitability'] = SUITABILITY_MESSAGES['no vegetation']
united_surfaces_with_area.loc[condition, 'reason'] = f'The roof area is less than {VEGETATION_AREA} m2, which is too small for vegetation installation.'

united_surfaces_with_area.drop(columns=['available_area_over_EGID'], inplace=True)

nbr_surfaces_tmp = united_surfaces_with_area.shape[0]
if nbr_surfaces_tmp != nbr_surfaces:
    logger.error('The number of roofs changed after setting the suitability for vegetation.' +
                 f' There is a difference of {nbr_surfaces_tmp - nbr_surfaces} surfaces compared to the original number.')
    
del available_surfaces_EGID, united_surfaces, area_tmp_gdf


logger.info('Check the buildings in industrial zones...')
industrial_zones.drop(columns=['NOM_ZONE', 'SOUS_ZONE', 'SURF_ZONE', 'SHAPE_AREA', 'SHAPE_LEN'], inplace=True)
industrial_zones.rename(columns={'OBJECTID': 'OBJECTID_IZ', 'NOM': 'NOM_ZONE', 'N_ZONE': 'NO_INDUSTRIAL_ZONE'}, inplace=True)

roofs_by_zone = gpd.sjoin(united_surfaces_with_area, industrial_zones, 
                        how='left', predicate='within', lsuffix='', rsuffix='industry')
roofs_by_zone.drop_duplicates(subset=united_surfaces_with_area.columns, inplace=True, ignore_index=True)

roofs_by_zone.loc[~roofs_by_zone.OBJECTID_IZ.isnull(), 'object_type'] = 'industrial space'

if roofs_by_zone.shape[0] != nbr_surfaces:
     logger.error('The number of roofs changed after setting the type of zone.' +
                 f' There is a difference of {roofs_by_zone.shape[0]-nbr_surfaces} surfaces compared to the original number.')


logger.info('Classify roofs less than 2 m2 as "too small" for solar panels...')

condition = (roofs_by_zone.SURFACE_TO <= SOLAR_ABS_MIN_AREA) & (roofs_by_zone.suitability == SUITABILITY_MESSAGES['ok'])
roofs_by_zone.loc[condition, 'suitability'] = SUITABILITY_MESSAGES['no solar']
roofs_by_zone.loc[condition, 'reason'] = f'The roof section is less than {SOLAR_ABS_MIN_AREA} m2, which is too small for solar panels.'

condition = (roofs_by_zone.SURFACE_TO <= SOLAR_ABS_MIN_AREA) & (roofs_by_zone.suitability != SUITABILITY_MESSAGES['ok'])
roofs_by_zone.loc[condition, 'suitability'] = SUITABILITY_MESSAGES['nothing']
roofs_by_zone.loc[condition, 'reason'] = roofs_by_zone.loc[condition, 'reason'] + f'The roof section is small than {SOLAR_ABS_MIN_AREA} m2, which is too small for solar panels.'


logger.info('Sort surfaces by slope...')

roofs_by_zone['roof_type'] = 'pitched'

roofs_by_zone.loc[
    (
        (roofs_by_zone.object_type != 'industrial space') & (roofs_by_zone['PENTE_MOYE'] < FLAT_VS_PITCHED_HOUSES)
    ) |
    (
        (roofs_by_zone.object_type == 'industrial space') & (roofs_by_zone['PENTE_MOYE'] < FLAT_VS_PITCHED_INDUSTRY)
    ),
    'roof_type'
] = 'flat'

# Resolving conflicts for roofs partially flat and pitched: attribution of the most frequent type over the EGID
# Better by area than occurence?
flat_occurences_egid = roofs_by_zone[roofs_by_zone.roof_type == 'flat'].value_counts(subset=['EGID'])
pitched_occurences_egid = roofs_by_zone[roofs_by_zone.roof_type == 'pitched'].value_counts(subset=['EGID'])
common_egid = flat_occurences_egid.index.intersection(pitched_occurences_egid.index)

logger.info(f'With a limit set at {FLAT_VS_PITCHED_HOUSES}° for houses and "skyscraper" and a limit set at {FLAT_VS_PITCHED_INDUSTRY}° for industrial '+
            f'buildings, there are {len(flat_occurences_egid)} flat roof buildings and {len(pitched_occurences_egid)} pitched roof buildings. ' +
            f'{len(common_egid)} buildings are in the two classes')
logger.info(f'Buildings with both flat and pitched roof segments are classified in the class with the greatest number of roof planes.')

egid_occurences = pd.concat([flat_occurences_egid.rename('flat'), pitched_occurences_egid.rename('pitched')], axis=1)
egid_occurences[egid_occurences.isna()] = 0
egid_occurences.reset_index(inplace=True)

flat_egids_list = egid_occurences.loc[egid_occurences.flat >= egid_occurences.pitched, 'EGID'].tolist()
pitched_egids_list = egid_occurences.loc[egid_occurences.flat < egid_occurences.pitched, 'EGID'].tolist()

roofs_by_zone['EGID_type'] = None
roofs_by_zone.loc[roofs_by_zone.EGID.isin(flat_egids_list), 'EGID_type'] = 'flat'
roofs_by_zone.loc[roofs_by_zone.EGID.isin(pitched_egids_list), 'EGID_type'] = 'pitched'

if any(roofs_by_zone.EGID_type.isnull()):
     logger.error('Some EGIDs were not classified as flat or pitched.')
     
del flat_occurences_egid, pitched_occurences_egid, egid_occurences


logger.info('Test the suitability of surfaces for solar installations...')

# Sum area available for solar over EGID
available_surfaces_EGID = roofs_by_zone[(roofs_by_zone.EGID != 0) & roofs_by_zone.suitability.isin([SUITABILITY_MESSAGES['ok'], SUITABILITY_MESSAGES['no vegetation']])]
area_tmp_gdf = available_surfaces_EGID.groupby(by='EGID')['SURFACE_TO'].sum().reset_index()
area_tmp_gdf.rename(columns={'SURFACE_TO': 'available_area_over_EGID'}, inplace=True)

# Take back the surfaces with EGID = 0 or that are unsuitable for solar
roofs_by_zone_with_area = roofs_by_zone.merge(area_tmp_gdf, on='EGID', how='left', suffixes=('_over_EGID', ''))
condition = roofs_by_zone_with_area.available_area_over_EGID.isnull() & (roofs_by_zone_with_area.suitability.isin([SUITABILITY_MESSAGES['ok'], SUITABILITY_MESSAGES['no vegetation']]))
roofs_by_zone_with_area.loc[condition, 'available_area_over_EGID'] = roofs_by_zone_with_area.loc[condition, 'SURFACE_TO']

roof_suitability_gdf = check_solar_suitability(roofs_by_zone_with_area)

all_roof_suitabilities_gdf=pd.concat([roof_suitability_gdf, missed_DIT_roofs], ignore_index=True)

nbr_surfaces_tmp = all_roof_suitabilities_gdf.shape[0]-missed_DIT_roofs.shape[0]
if nbr_surfaces_tmp != nbr_surfaces:
     logger.error('The number of surfaces changed after testing the suitability of roofs for solar installation.' +
                 f' There is a difference of {nbr_surfaces_tmp-nbr_surfaces} surfaces compared to the original number.')
     
del available_surfaces_EGID, roof_suitability_gdf, roofs_by_zone


logger.info('Indicate buildings in heritage zones...')

heritage_ensemble.rename(columns={'OBJECTID': 'OBJECTID_heritage', 'N_CLASSEME': 'NO_CLASSE'}, inplace=True, errors='raise')
heritage_classement.rename(columns={'OBJECTID': 'OBJECTID_heritage'}, inplace=True, errors='raise')
heritage = pd.concat(
    [heritage_classement[['OBJECTID_heritage', 'NO_CLASSE', 'geometry']], 
      heritage_ensemble[['OBJECTID_heritage', 'NO_CLASSE', 'geometry']]]
)
heritage['geometry'] = heritage.geometry.buffer(3)

final_roofs_gdf = gpd.sjoin(
    all_roof_suitabilities_gdf, heritage, how='left', predicate='within', lsuffix='roof', rsuffix='heritage'
)

# Set the heritage status for the objects with EGID.
heritage_egids = final_roofs_gdf.loc[final_roofs_gdf['NO_CLASSE'].notna(),'EGID'].unique().tolist()
heritage_egids.remove(0)    # Avoid all the objects without EGID to be considered as heritage.

heritage_condition = final_roofs_gdf.EGID.isin(heritage_egids)

final_roofs_gdf = set_heritage_status(final_roofs_gdf, condition)

# Set the heritage status for objects without EGID.
heritage_no_EGID_objects = final_roofs_gdf.loc[(final_roofs_gdf['NO_CLASSE'].notna()) & (final_roofs_gdf['EGID']==0), 'OBJECTID_OCEN'].unique().tolist()
heritage_condition = final_roofs_gdf.OBJECTID_OCEN.isin(heritage_no_EGID_objects)
final_roofs_gdf = set_heritage_status(final_roofs_gdf, heritage_condition)


final_roofs_gdf.drop_duplicates(subset=['OBJECTID_OCEN', 'OBJECTID_DIT'], ignore_index = True, inplace=True)

if final_roofs_gdf.shape[0] != nbr_surfaces + missed_DIT_roofs.shape[0]:
    logger.error('The number of roofs changed after join the heritage geodata.' +
                 f' There is a difference of {final_roofs_gdf.shape[0]+nbr_surfaces+missed_DIT_roofs.shape[0]} surfaces compared to the original number.')

final_roofs_gdf.loc[final_roofs_gdf.suitability.isnull(), 'suitability'] == 'suitable for both uses.'

final_roofs_gdf = final_roofs_gdf[['OBJECTID_DIT', 'OBJECTID_OCEN', 'suitability', 'reason', 
                                   'EGID', 'object_type', 'roof_type', 'TYPE_SURFA', 'OBJECTID_IZ', 'OBJECTID_heritage',
                                   'ORIENTATIO', 'PENTE_MOYE', 'IRR_MOYENN', 'SURFACE_TO', 'geometry']]

logger.info('Save file...')
final_roofs_gdf.to_file(FILEPATH, layer='attributes_filtered_roofs', index=False)

logger.success(f'Done! The results were written in the geopackage {FILEPATH}, in the layer "attributes_filtered_roofs".')