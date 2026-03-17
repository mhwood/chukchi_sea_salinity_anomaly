

import os
import numpy as np
import matplotlib.pyplot as plt
import eccoseas as es

data_folder = '/Users/mike/Documents/Research/Projects/Ocean_Modeling/ECCO/LLC270_Files/era_xx'

years = [2023, 2024]

for year in years:

    if year%4==0:
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



