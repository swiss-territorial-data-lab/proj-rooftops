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


# Define functions -------------------------

def prepare_solar_check(gdf):
    '''
    Calculate the total surface of roof per EGID

    - gdf: geodataframe of the roofs with the columns EGID and SURFACE_TO.
    return: the same geodataframe, but with an attribute "tot_surface_EGID"
    '''

    gdf_by_egid=gdf.groupby(by=['EGID'])['SURFACE_TO'].sum().reset_index()
    gdf_by_egid.rename(columns={'SURFACE_TO':'tot_surface_EGID'}, inplace=True)
    gdf_by_egid_and_zone=gdf.merge(gdf_by_egid, on='EGID')

    return gdf_by_egid_and_zone

def check_solar_suitability(gdf_by_egid_and_zone, messages, solar_house=0, solar_industry=0):
    '''
    Do the if-else statement to test if there is enough space for a solar installation.
   
    - gdf_by_egid_and_zone: geodataframe of the roofs with the columns tot_surface_EGID, NO_INDUSTRIAL_ZONE, suitability, and reason.
    - messages: dictionary of the message to indicate the lack of suitability with the keys "no vegetation", "no solar" and "nothing".
    - solar_house: limit in m2 to which it becomes worth to install a solar installation on a house.
    - solar_industry: limit in m2 to which it becomes worth to install a solar installation on an industry.
    return: the same geodataframe, but with the attributes "suitability" and "reason" updated with the solar informations
    '''

    min_area=min([solar_house, solar_industry])

    solar_suitability=[]
    solar_reason=[]

    for roof in gdf_by_egid_and_zone.itertuples():

        # The building does not have any other contraindications.
        if roof.suitability==None:
            # The roof has at least the min area
            if roof.tot_surface_EGID<min_area:
                solar_suitability.append(messages['no solar'])
                solar_reason.append(f'The roof area over the EGID is under {min_area} m2 which is too small for solar installation.')
                continue

            # The building is NOT industrial
            if not isinstance(roof.NO_INDUSTRIAL_ZONE, str):

                # The surface is too small for solar panels
                if roof.tot_surface_EGID < solar_house:
                    solar_suitability.append(messages['no solar'])
                    solar_reason.append(f'The roof area over the EGID is under {solar_house} m2 which is too small for solar installation.')
                    continue

                # The surface is ok for solar panels.
                else:
                    solar_suitability.append(None)
                    solar_reason.append(None)
                    continue
            
            # The building is industrial.
            else:

                # The surface is too small for solar panels
                if roof.tot_surface_EGID < solar_industry:
                    solar_suitability.append(messages['no solar'])
                    solar_reason.append(f'The roof area over the EGID is under {solar_industry} m2 which is too small for solar installation.')
                    continue

                # The surface is ok for solar panels.
                else:
                    solar_suitability.append(None)
                    solar_reason.append(None)
                    continue

        # The building is already unsuited for vegetation.
        if roof.suitability==messages['no vegetation']:

            # The roof has at least the min area
            if roof.tot_surface_EGID<min_area:
                solar_suitability.append(messages['nothing'])
                solar_reason.append(roof.reason + 
                                    f' The roof area over the EGID is under {min_area} m2 which is too small for solar installation.')
                continue

            # The building is NOT industrial
            if not isinstance(roof.NO_INDUSTRIAL_ZONE, str):

                # The surface is too small for solar panels
                if roof.tot_surface_EGID < solar_house:
                    solar_suitability.append(messages['nothing'])
                    solar_reason.append(roof.reason + 
                                        f' The roof area over the EGID is under {solar_house} m2 which is too small for solar installation.')
                    continue

                # The surface is ok for solar panels.
                else:
                    solar_suitability.append(roof.suitability)
                    solar_reason.append(roof.reason)
                    continue
            
            # The building is industrial.
            else:

                # The surface is too small for solar panels
                if roof.tot_surface_EGID < solar_industry:
                    solar_suitability.append(messages['nothing'])
                    solar_reason.append(roof.reason + 
                                        f' The roof area is under {solar_industry} m2 which is too small for solar installation.')
                    continue

                # The surface is ok for solar panels.
                else:
                    solar_suitability.append(roof.suitability)
                    solar_reason.append(roof.reason)
                    continue
        else:
            solar_suitability.append(roof.suitability)
            solar_reason.append(roof.reason)

    gdf_by_egid_and_zone['suitability']=solar_suitability
    gdf_by_egid_and_zone['reason']=solar_reason

    return gdf_by_egid_and_zone


# Define constants --------------------------

WORKING_DIRECTORY=cfg['working_directory']
INPUT_FILES=cfg['input_files']
PARAMETERS=cfg['parameters']

ROOFS=INPUT_FILES['roofs']
SOLAR_SURFACES=INPUT_FILES['potential_solar_surfaces']
HERITAGE_ENSEMBLE=INPUT_FILES['heritage']['ensembles']
HERITAGE_CLASSEMENT=INPUT_FILES['heritage']['classement']
INDUSTRIAL_ZONES=INPUT_FILES['industrial_zones']

SOLAR_ABS_MIN_AREA=PARAMETERS['solar_absolute_minimum_area']
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

SUITABILITY_MESSAGES={'no vegetation': 'unsuitable for vegetation', 'no solar': 'unsuitable for solar installations', 
                      'nothing': 'not suitable for valorization', 'uncertain': 'unsure'}

os.chdir(WORKING_DIRECTORY)
FILEPATH=os.path.join(fct_misc.ensure_dir_exists('processed/roofs'), 'roofs.gpkg')

logger.info('Reading input files...')

roofs=gpd.read_file(ROOFS)
solar_surfaces=gpd.read_file(SOLAR_SURFACES)
heritage_ensemble=gpd.read_file(HERITAGE_ENSEMBLE)
heritage_classement=gpd.read_file(HERITAGE_CLASSEMENT)
industrial_zones=gpd.read_file(INDUSTRIAL_ZONES)

roofs=fct_misc.test_valid_geom(roofs[['OBJECTID', 'geometry', 'SURFACE_TO', 'PENTE_MOY', 'EGID']], correct=True, gdf_obj_name='DIT roofs')
solar_surfaces=fct_misc.test_valid_geom(
    solar_surfaces[['OBJECTID', 'EGID', 'TYPE_SURFA', 'ID_SURFACE', 'ORIENTATIO', 'PENTE_MOYE', 'IRR_MOYENN', 'SURFACE_TO', 'geometry']], 
    correct=True, gdf_obj_name='solar surfaces'
)


logger.info('Uniting the roofs as defined by the DIT and the OCEN...')

solar_surfaces['area']=round(solar_surfaces.area, 3)
joined_surfaces=gpd.sjoin(
    roofs[['OBJECTID', 'geometry']], solar_surfaces, how='right', predicate='intersects', lsuffix='DIT', rsuffix='OCEN'
)
roofs['geom_DIT']=roofs.geometry
joined_surfaces_with_area=joined_surfaces.merge(roofs[['OBJECTID', 'geom_DIT']], how='left', left_on='OBJECTID_DIT', right_on='OBJECTID')

intersecting_area=[]
for (geom1, geom2) in zip(joined_surfaces_with_area.geom_DIT.values.tolist(), joined_surfaces_with_area.geometry.values.tolist()):
    if geom1 is not None:
        intersecting_area.append(round(geom1.intersection(geom2).area/geom2.area, 3))
    else:
        intersecting_area.append(None)

joined_surfaces_with_area['intersecting_area']=intersecting_area
joined_surfaces_with_area.drop(columns=['geom_DIT', 'index_DIT'], inplace=True)
joined_surfaces_with_area.sort_values(by=['intersecting_area'], ascending=False, inplace=True)

united_surfaces=joined_surfaces_with_area.drop_duplicates(subset='OBJECTID_OCEN', ignore_index = True)

captured_dit_id=united_surfaces['OBJECTID_DIT'].unique().tolist()
missed_DIT_roofs=roofs[~roofs['OBJECTID'].isin(captured_dit_id)].copy()

united_surfaces.loc[united_surfaces['intersecting_area'] < 0.75, 'OBJECTID_DIT']=NaN

missed_DIT_roofs.rename(columns={'OBJECTID': 'OBJECTID_DIT', 'PENTE_MOY':'PENTE_MOYE'}, inplace=True)
missed_DIT_roofs.drop(columns=['geom_DIT'], inplace=True)
all_surfaces=pd.concat([united_surfaces, missed_DIT_roofs], ignore_index=True)

nbr_surfaces=all_surfaces.shape[0]
logger.info(f'There are {nbr_surfaces} roof shapes for {len(all_surfaces.EGID.unique().tolist())} EGIDs.')

del roofs, solar_surfaces, joined_surfaces, united_surfaces


logger.info('Setting suitability for vegetation based on roof slope...')
all_surfaces['suitability']=[
    SUITABILITY_MESSAGES['no vegetation']
    if slope > VEGETATION_INCLINATION else None
    for slope in all_surfaces['PENTE_MOYE'].to_numpy()
]
all_surfaces['reason']=[
    'The slope is too steep for vegetation.'
    if slope > VEGETATION_INCLINATION else None
    for slope in all_surfaces['PENTE_MOYE'].to_numpy()
]

logger.info('Separating roofs from parkings and other covers...')
roofs_to_process=all_surfaces[all_surfaces.EGID!=0]
other_surfaces=all_surfaces[all_surfaces.EGID==0]

if any(all_surfaces.EGID.isnull()):
    logger.error('There are some roofs with a null EGID that are not processed to the end.')


logger.info('Testing the area left on each surface for vegetation...')

roofs_accepting_vege=roofs_to_process[roofs_to_process.suitability.isnull()]
area_tmp_gdf=roofs_accepting_vege.groupby(by='EGID')['SURFACE_TO'].sum().reset_index()

area_tmp_gdf['suitability']=[
    SUITABILITY_MESSAGES['no vegetation'] if surface < VEGETATION_AREA else None for surface in area_tmp_gdf.SURFACE_TO.to_numpy()
]
area_tmp_gdf['reason']=[
    f'The roof area is under {VEGETATION_AREA} m2 which is too small for vegetation.' 
    if surface < VEGETATION_AREA else None for surface in area_tmp_gdf.SURFACE_TO.to_numpy()
]

# Bring the result back into the global dataframe
roofs_with_vegetation_suitability=roofs_to_process.merge(area_tmp_gdf, how='left', on='EGID', suffixes=('', '_vege'))

roofs_with_vegetation_suitability.loc[
    roofs_with_vegetation_suitability.suitability_vege == SUITABILITY_MESSAGES['no vegetation'], 'suitability'
] = SUITABILITY_MESSAGES['no vegetation']
roofs_with_vegetation_suitability.loc[
    roofs_with_vegetation_suitability.suitability_vege == SUITABILITY_MESSAGES['no vegetation'], 'reason'
] = roofs_with_vegetation_suitability.loc[
    roofs_with_vegetation_suitability.suitability_vege == SUITABILITY_MESSAGES['no vegetation'], 'reason_vege'
    ]

roofs_with_vegetation_suitability.drop(columns=['SURFACE_TO_vege', 'suitability_vege', 'reason_vege'], inplace=True)

other_surfaces.loc[other_surfaces.suitability.isnull(), 'reason']=[
    'This surface is not a roof.' for surface in other_surfaces[other_surfaces.suitability.isnull()].SURFACE_TO.to_numpy()
]
other_surfaces.loc[other_surfaces.suitability.isnull(), 'suitability']=[
    SUITABILITY_MESSAGES['no vegetation'] for surface in other_surfaces[other_surfaces.suitability.isnull()].SURFACE_TO.to_numpy()
]

nbr_surfaces_tmp=roofs_with_vegetation_suitability.shape[0] + other_surfaces.shape[0]
if nbr_surfaces_tmp!=nbr_surfaces:
    logger.error('The number of roofs changed after setting the suitability for vegetation.' +
                 f' There is a difference of {nbr_surfaces_tmp-nbr_surfaces} surfaces compared to the original number.')
    
del roofs_accepting_vege, all_surfaces, roofs_to_process, area_tmp_gdf


logger.info('Determining which surfaces are parts of industrial buildings...')
industrial_zones.drop(columns=['NOM_ZONE', 'SOUS_ZONE', 'SURF_ZONE', 'SHAPE_AREA', 'SHAPE_LEN'], inplace=True)
industrial_zones.rename(columns={'OBJECTID': 'OBJECTID_IZ', 'NOM': 'NOM_ZONE', 'N_ZONE': 'NO_INDUSTRIAL_ZONE'}, inplace=True)

roofs_by_zone=gpd.sjoin(roofs_with_vegetation_suitability, industrial_zones, 
                        how='left', predicate='within', lsuffix='', rsuffix='industy')
roofs_by_zone.drop_duplicates(subset=roofs_with_vegetation_suitability.columns, inplace=True, ignore_index=True)

other_surfaces_by_zone=gpd.sjoin(other_surfaces, industrial_zones, 
                        how='left', predicate='within', lsuffix='', rsuffix='industy')
other_surfaces_by_zone.drop_duplicates(subset=other_surfaces.columns, inplace=True, ignore_index=True)

nbr_surfaces_tmp=roofs_by_zone.shape[0] + other_surfaces_by_zone.shape[0]
if nbr_surfaces_tmp != nbr_surfaces:
     logger.error('The number of roofs changed after setting the type of zone.' +
                 f' There is a difference of {nbr_surfaces_tmp-nbr_surfaces} surfaces compared to the original number.')


logger.info('Marking the roofs less than 2 m2 as too small...')
large_roofs_by_zone=roofs_by_zone[roofs_by_zone['SURFACE_TO']>SOLAR_ABS_MIN_AREA].copy()
small_roofs_by_zone=roofs_by_zone[roofs_by_zone['SURFACE_TO']<=SOLAR_ABS_MIN_AREA].copy()

small_roofs_by_zone.loc[:,'suitability']=[
    SUITABILITY_MESSAGES['no solar']
    if suitability==None else SUITABILITY_MESSAGES['nothing']
    for suitability in small_roofs_by_zone.suitability.to_numpy()
]
small_roofs_by_zone.loc[:,'reason']=[
    f'The roof section is small than {SOLAR_ABS_MIN_AREA} m2, which is too small for a solar panel.'
    if reason==None else reason + f'The roof section is small than {SOLAR_ABS_MIN_AREA} m2, which is too small for a solar panel.'
    for reason in small_roofs_by_zone.reason.to_numpy()
]


logger.info('Sorting the surfaces by slope...')

flat_roofs_by_zone_tmp=large_roofs_by_zone[
    (
        (large_roofs_by_zone['NO_INDUSTRIAL_ZONE'].isnull()) & (large_roofs_by_zone['PENTE_MOYE'] < FLAT_VS_PITCHED_HOUSES)
    ) |
    (
        (~large_roofs_by_zone['NO_INDUSTRIAL_ZONE'].isnull()) & (large_roofs_by_zone['PENTE_MOYE'] < FLAT_VS_PITCHED_INDUSTRY)
    )
]

pitched_roofs_by_zone_tmp=large_roofs_by_zone[
    (
        (large_roofs_by_zone['NO_INDUSTRIAL_ZONE'].isnull()) & (large_roofs_by_zone['PENTE_MOYE'] >= FLAT_VS_PITCHED_HOUSES)
    ) |
    (
        (~large_roofs_by_zone['NO_INDUSTRIAL_ZONE'].isnull()) & (large_roofs_by_zone['PENTE_MOYE'] >= FLAT_VS_PITCHED_INDUSTRY)
    )
]

# Resolving conflicts for roofs partially flat and pitched
# Better by area than occurence?
flat_occurences_egid=flat_roofs_by_zone_tmp.value_counts(subset=['EGID'])
pitched_occurences_egid=pitched_roofs_by_zone_tmp.value_counts(subset=['EGID'])
common_egid=flat_occurences_egid.index.intersection(pitched_occurences_egid.index)

logger.info(f'With a limit at {FLAT_VS_PITCHED_HOUSES}° for houses and "skyscraper" and a limit at {FLAT_VS_PITCHED_INDUSTRY}° for industrial '+
            f'buildings, there are {len(flat_occurences_egid)} flat roof buildings and {len(pitched_occurences_egid)} pitched roof buildings. ' +
            f'{len(common_egid)} buildings are in the two classes')
logger.info(f'Buildings are suppressed on the side where they have the least occurences.')

egid_occurences=pd.concat([flat_occurences_egid, pitched_occurences_egid], axis=1)
egid_occurences[egid_occurences.isna()]=0
egid_occurences.reset_index(inplace=True)

flat_egids=egid_occurences.loc[egid_occurences[0] >= egid_occurences[1], 'EGID'].tolist()
pitched_egids=egid_occurences.loc[egid_occurences[0] < egid_occurences[1], 'EGID'].tolist()

flat_roofs_by_zone=large_roofs_by_zone[large_roofs_by_zone.EGID.isin(flat_egids)]
pitched_roofs_by_zone=large_roofs_by_zone[large_roofs_by_zone.EGID.isin(pitched_egids)]

flat_other_surfaces_by_zone=other_surfaces_by_zone[
    (
        (other_surfaces_by_zone['NO_INDUSTRIAL_ZONE'].isnull()) & (other_surfaces_by_zone['PENTE_MOYE'] < FLAT_VS_PITCHED_HOUSES)
    ) |
    (
        (~other_surfaces_by_zone['NO_INDUSTRIAL_ZONE'].isnull()) & (other_surfaces_by_zone['PENTE_MOYE'] < FLAT_VS_PITCHED_INDUSTRY)
    )
]

pitched_other_surfaces_by_zone=other_surfaces_by_zone[
    (
        (other_surfaces_by_zone['NO_INDUSTRIAL_ZONE'].isnull()) & (other_surfaces_by_zone['PENTE_MOYE'] >= FLAT_VS_PITCHED_HOUSES)
    ) |
    (
        (~other_surfaces_by_zone['NO_INDUSTRIAL_ZONE'].isnull()) & (other_surfaces_by_zone['PENTE_MOYE'] >= FLAT_VS_PITCHED_INDUSTRY)
    )
]

nbr_surfaces_tmp = flat_roofs_by_zone.shape[0] + flat_other_surfaces_by_zone.shape[0] + \
                    pitched_roofs_by_zone.shape[0] + pitched_other_surfaces_by_zone.shape[0] + small_roofs_by_zone.shape[0]
if nbr_surfaces_tmp != nbr_surfaces:
     logger.error('The number of surfaces changed after setting separating between flat and pitched surfaces.' +
                 f' There is a difference of {nbr_surfaces_tmp-nbr_surfaces} surfaces compared to the original number.')
     
del roofs_by_zone, other_surfaces, other_surfaces_by_zone
del flat_occurences_egid, pitched_occurences_egid, egid_occurences
del flat_roofs_by_zone_tmp, pitched_roofs_by_zone_tmp

logger.info('Testing the suitability of surfaces for solar installations...')

flat_roofs_by_egid_and_zone=prepare_solar_check(flat_roofs_by_zone)
flat_roofs_by_egid_and_zone=check_solar_suitability(flat_roofs_by_egid_and_zone, SUITABILITY_MESSAGES, 
                        solar_house=FLAT_SOLAR_HOUSES, solar_industry=FLAT_SOLAR_INDUSTRY)
flat_other_surfaces_by_zone['tot_surface_EGID']=flat_other_surfaces_by_zone['SURFACE_TO']
flat_other_surfaces_by_egid_and_zone=check_solar_suitability(flat_other_surfaces_by_zone, SUITABILITY_MESSAGES,
                                                             FLAT_SOLAR_HOUSES, FLAT_SOLAR_INDUSTRY)

pitched_roofs_by_egid_and_zone=prepare_solar_check(pitched_roofs_by_zone)
pitched_roofs_by_egid_and_zone=check_solar_suitability(pitched_roofs_by_egid_and_zone, SUITABILITY_MESSAGES, 
                        solar_house=PITCHED_SOLAR_HOUSES, solar_industry=PITCHED_SOLAR_INDUSTRY)
pitched_other_surfaces_by_zone['tot_surface_EGID']=pitched_other_surfaces_by_zone['SURFACE_TO']
ptiched_other_surfaces_by_egid_and_zone=check_solar_suitability(pitched_other_surfaces_by_zone, SUITABILITY_MESSAGES,
                                                             PITCHED_SOLAR_HOUSES, PITCHED_SOLAR_INDUSTRY)

surfaces_by_egid_and_zone=pd.concat(
    [
        flat_roofs_by_egid_and_zone, flat_other_surfaces_by_egid_and_zone, 
        pitched_roofs_by_egid_and_zone, ptiched_other_surfaces_by_egid_and_zone,
        small_roofs_by_zone
    ], ignore_index=True
)

nbr_surfaces_tmp = surfaces_by_egid_and_zone.shape[0]
if nbr_surfaces_tmp != nbr_surfaces:
     logger.error('The number of surfaces changed after testing the suitability of roofs for solar installation.' +
                 f' There is a difference of {nbr_surfaces_tmp-nbr_surfaces} surfaces compared to the original number.')

del small_roofs_by_zone


logger.info('Indicating buildings in heritage zones...')

heritage_ensemble.rename(columns={'N_CLASSEME': 'NO_CLASSE'}, inplace=True, errors='raise')
heritage=pd.concat(
    [heritage_classement[['OBJECTID', 'NO_CLASSE', 'geometry']], 
      heritage_ensemble[['OBJECTID', 'NO_CLASSE', 'geometry']]]
)
heritage['geometry']= heritage.geometry.buffer(3)

surfaces_with_heritage_info=gpd.sjoin(
    surfaces_by_egid_and_zone, heritage, how='left', predicate='within', lsuffix='roof', rsuffix='heritage'
)
heritage_egids=surfaces_with_heritage_info.loc[surfaces_with_heritage_info['NO_CLASSE'].notna(),'EGID'].unique().tolist()
heritage_egids.remove(0)

surfaces_with_heritage_info.loc[
    (surfaces_with_heritage_info.EGID.isin(heritage_egids)) & (surfaces_with_heritage_info.suitability != SUITABILITY_MESSAGES['nothing']), 
    'suitability'
]=SUITABILITY_MESSAGES['uncertain']
surfaces_with_heritage_info.loc[
    (surfaces_with_heritage_info.suitability == SUITABILITY_MESSAGES['uncertain']) & (surfaces_with_heritage_info.reason.isnull()), 
    'reason'
]='This EGID is part of an heritage building'
surfaces_with_heritage_info.loc[
    (surfaces_with_heritage_info.suitability == SUITABILITY_MESSAGES['uncertain']) & (~surfaces_with_heritage_info.reason.isnull()), 
    'reason'
]=surfaces_with_heritage_info.loc[
        (surfaces_with_heritage_info.suitability == SUITABILITY_MESSAGES['uncertain']) & (~surfaces_with_heritage_info.reason.isnull()), 
        'reason'
    ] + ' This EGID is part of an heritage building.'

surfaces_with_heritage_info.drop_duplicates(subset=['OBJECTID_OCEN', 'OBJECTID_DIT'], ignore_index = True, inplace=True)

if surfaces_with_heritage_info.shape[0]!=nbr_surfaces:
    logger.error('The number of roofs changed after the join the heritage geodata.' +
                 f' There is a difference of {surfaces_with_heritage_info.shape[0]-nbr_surfaces} surfaces compared to the original number.')
    

logger.info('Saving file...')
surfaces_with_heritage_info.to_file(FILEPATH, layer='attributes_filtered_roofs')

logger.success(f'Done! The results were written in the geopackage {FILEPATH}, in the layer "attributes_filtered_roofs".')

