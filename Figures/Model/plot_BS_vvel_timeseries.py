
import os
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc4


def read_modeled_DBO_section(project_folder, var_name, experiment, dbo_section):

    #'DBO Sections',
    dbo_file_name = os.path.join(project_folder,'Data', 'Model', 'DBO Sections',
                                 var_name + '_DBO_Sections_' + experiment + '.nc')
    ds = nc4.Dataset(dbo_file_name)

    grp = ds.groups[dbo_section]
    dec_yrs = grp.variables['dec_yrs'][:]
    Z = grp.variables['depth'][:]
    var_grid = grp.variables[var_name][:,:,:]

    ds.close()
    return dec_yrs, Z, var_grid

def plot_vvel_timeseries(project_folder, experiments, vvel_grids, dbo_section):
    fig = plt.figure(figsize=(9,5))
    plt.style.use('dark_background')

    for experiment in experiments:
        vvel_grid = vvel_grids[experiment]
        vvel_timeseries = np.nanmean(vvel_grid[:,0,:], axis=-1)

        plt.plot(dec_yrs, vvel_timeseries, label=experiment)

    plt.title(f'Average Northward Velocity Timeseries - {dbo_section} Section')
    plt.ylabel('Northward Velocity (m/s)')

    plt.grid(linestyle='--', alpha=0.5)
    plt.legend()

    output_file = os.path.join(project_folder, 'Figures',
                               f'vvel_timeseries_{dbo_section}.png')
    plt.savefig(output_file)
    plt.close(fig)




project_folder = '/Users/mike/Documents/Research/Projects/Chukchi Sea/'

var_name = 'Vvel'
dbo_section = "BS"

experiments = ['control', 'daily_hydrology', 'prescribed_seaice']

vvel_grids = {}

for experiment in experiments:
    dec_yrs, Z, vvel_grid = read_modeled_DBO_section(project_folder, var_name, experiment, dbo_section)
    vvel_grids[experiment] = vvel_grid

plot_vvel_timeseries(project_folder, experiments, vvel_grids, dbo_section)













