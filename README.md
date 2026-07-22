# dynamical_changes_mortality
Scripts for study on the heat-related mortality associated with dynamical changes
Scripts are provided for the code developed for this study. Some data must be preprocesed from previous studies, or could not be uploaded because the files were too large. 

Preprocessing using CDO:
# CDO code
cdo delfeb29 whole temp series
cdo ydaysub to compute clim
clim lowpass 10 to smooth clim
cdo trend to compute slope and intercept on anomalies
