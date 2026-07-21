# dynamical_changes_mortality
Scripts for study on the heat-related mortality associated with dynamical changes

Preprocessing using CDO:
# CDO code
cdo delfeb29 whole temp series
cdo ydaysub to compute clim
clim lowpass 10 to smooth clim
cdo trend to compute slope and intercept on anomalies
