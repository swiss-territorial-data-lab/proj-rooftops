# proj-rooftops

This branch serves to assess the quality of the results provided by flai compared to our ground truth.
The script `assess_flai.py`and the functions `get_fractional_sets` and `get_metrics` are mostly taken from the GitHub branch `ch/SAM`. Some modifications were performed in order to allow a one-to-many cardinality between the predictions and the labels. In addition, the use of the attribute `value` is only adapted to the specific case of the SAM algorithm and had to be changed.