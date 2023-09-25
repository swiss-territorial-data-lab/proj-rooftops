# proj-rooftops

### Filtering of the roof parameters

**Data**: This workflow is based on the following layers, available in the [SITG catalog](http://ge.ch/sitg/sitg_catalog/sitg_donnees). <br>
- CAD_BATIMENT_HORSOL_TOIT.shp: Roof areas of above-ground buildings.
- OCEN_SOLAIRE_ID_SURFACE_BASE.shp: roofs, sheds and parkings.
- FTI_PERIMETRE.shp: perimeters of the industrial zones managed by the Foundation for Industrial Lands of Geneva.
- DPS_ENSEMBLE.shp & DPS_CLASSEMENT.shp: architectural and landscape surveys of the canton, archaeological and archival research sites, and scientific inventories. Listed buildings in accordance with the cantonal law on the protection of monuments and sites.

**Requirements**:
- There are no hardware or software requirements.
- Everything was tested with Python 3.11.
- The necessary libraries can be installed from the file `requirements.txt`.

The path to the config file is hard-coded at the start of each script.

## Workflow

<figure align="center">
<image src="img\attribute_filtering_flow.jpeg" alt="Diagram of the methodology" style="width:60%;">
<figcaption align="center">Diagram of the criteria applied to determine the roof suitability for vegetation and solar panels.</figcaption> 
</figure>

Everything is executed in one script. 

```
python filter_by_attributes.py
```

### Collaboration with flai

This branch serves to assess the quality of the results provided by flai compared to our ground truth.
The script `assess_flai.py`and the functions `get_fractional_sets` and `get_metrics` are mostly taken from the GitHub branch `ch/SAM`. Some modifications were performed in order to allow a one-to-many cardinality between the predictions and the labels. In addition, the use of the attribute `value` is only adapted to the specific case of the SAM algorithm and had to be changed.
