

import os
import numpy as np
import matplotlib.pyplot as plt
import eccoseas as es

data_folder = '/Users/mhwood/Documents/Research/Projects/Ocean_Modelling/ECCO/LLC270_Files/era_xx'

years = np.arange(1992,2025).tolist()

for year in years:

    print('Working on year ' + str(year))

    if year%4==0 or year==2013:
        days = 366
    else:
        days = 365


    grid = np.fromfile(data_folder+'/EIG_rain_plus_ECCO_v4r1_ctrl_'+str(year), '>f4').reshape((days*4,256,512))

    if year==years[0]:
        precip = grid[:365*4,:,:]
    else:
        precip += grid[:365*4,:,:]

precip /= len(years)

# Save the climatology
output_file = os.path.join(data_folder, 'EIG_rain_plus_ECCO_v4r1_ctrl_climatology')
precip.ravel(order='C').astype('>f4').tofile(output_file)



