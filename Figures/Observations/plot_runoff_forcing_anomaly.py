
import os
import numpy as np
import netCDF4 as nc4
import matplotlib.pyplot as plt
from pyproj import Transformer
from scipy.interpolate import griddata
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def reproject_points(polygon_array,inputCRS,outputCRS,x_column=0,y_column=1,run_test = True):

    transformer = Transformer.from_crs('EPSG:' + str(inputCRS), 'EPSG:' + str(outputCRS))

    # There seems to be a serious problem with pyproj
    # The x's and y's are mixed up for these transformations
    #       For 4326->3413, you put in (y,x) and get out (x,y)
    #       Foe 3413->4326, you put in (x,y) and get out (y,x)
    # Safest to run check here to ensure things are outputting as expected with future iterations of pyproj

    if inputCRS == 4326 and outputCRS == 3413:
        x2, y2 = transformer.transform(polygon_array[:,y_column], polygon_array[:,x_column])
        x2 = np.array(x2)
        y2 = np.array(y2)
    elif inputCRS == 3413 and outputCRS == 4326:
        y2, x2 = transformer.transform(polygon_array[:, x_column], polygon_array[:, y_column])
        x2 = np.array(x2)
        y2 = np.array(y2)
    elif str(inputCRS)[:3] == '326' and outputCRS == 3413:
        x2, y2 = transformer.transform(polygon_array[:,y_column], polygon_array[:,x_column])
        x2 = np.array(x2)
        y2 = np.array(y2)
        run_test = False
    elif str(inputCRS)[:3] == '326' and outputCRS == 4326:
        y2, x2 = transformer.transform(polygon_array[:, x_column], polygon_array[:, y_column])
        x2 = np.array(x2)
        y2 = np.array(y2)
        run_test = False
    elif inputCRS == 4326 and str(outputCRS)[:3] == '326':
        x2, y2 = transformer.transform(polygon_array[:, y_column], polygon_array[:, x_column])
        x2 = np.array(x2)
        y2 = np.array(y2)
        run_test = False
    else:
        raise ValueError('Reprojection with this epsg is not safe - no test for validity has been implemented')

    output_polygon=np.copy(polygon_array)
    output_polygon[:,x_column] = x2
    output_polygon[:,y_column] = y2
    return output_polygon

def read_model_grid(project_folder):
    ds = nc4.Dataset(os.path.join(project_folder, 'Data','Model', 'Chukchi_Sea_grid.nc'))
    XC = ds.variables['XC'][:,:]
    YC = ds.variables['YC'][:,:]
    Depth = ds.variables['Depth'][:,:]
    rA = ds.variables['rA'][:,:]
    ds.close()
    return XC, YC, Depth, rA

def read_Fekete_runoff_data(project_folder):

    grid = np.fromfile(os.path.join(project_folder, 'Model','input', 'runoff-2d-Fekete-coastal.bin'), '>f4').reshape((12,720,960))
    # convert m/s to mm/day
    # grid *= 86400  # Convert from m/s to mm/day

    return(grid)

def read_Blaskey_runoff_data(project_folder):
    ds = nc4.Dataset(os.path.join(project_folder, 'Data', 'Observations','River Discharge',
                                  'Alaska_River_discharge_in_Chukchi_Sea_domain.nc'))
    runoff = ds.variables['runoff'][:,:]
    ds.close()

    # plt.plot(runoff[15,:])
    # plt.show()

    return runoff

def read_reconstructed_runoff_data(project_folder):

    grid_2023 = np.fromfile(os.path.join(project_folder, 'Model','input',
                                         'runoff-2d-Fekete-coastal_2023'), '>f4').reshape((365, 720, 960))
    grid_2024 = np.fromfile(os.path.join(project_folder, 'Model', 'input',
                                         'runoff-2d-Fekete-coastal_2024'), '>f4').reshape((366, 720, 960))

    return(grid_2023, grid_2024)

def plot_summer_runoff(project_folder, runoff_fekete_climatology, runoff_blaskey):#, runoff_2023, runoff_2024):

    fig = plt.figure(figsize=(8, 10))

    plot_height = 8

    gs = GridSpec(4, 2, left=0.12, right=0.92, bottom=0.05, top=0.95, hspace=0.2)

    #####################################################################################
    # Timeseries from Fekete et al 2002

    ax = plt.subplot(gs[0, :])
    days = np.arange(1, 400, 0.25)  # Days of the year

    months = np.arange(1, 13)+0.5

    runoff_timeseries_climatology = np.sum(runoff_fekete_climatology, axis=(1, 2))
    runoff_timeseries_alaska = np.sum(runoff_fekete_climatology[:, :, 580:], axis=(1, 2))
    runoff_timeseries_russia = np.sum(runoff_fekete_climatology[:, :, :580], axis=(1, 2))
    plt.plot(months, runoff_timeseries_climatology,
             label='Total (Mean: '+f'{np.mean(runoff_timeseries_climatology):.0f})' + ' m$^3$/s)',
             color='orange')
    plt.plot(months, runoff_timeseries_alaska, label='Alaska (Mean: '+f'{np.mean(runoff_timeseries_alaska):.0f})' + ' m$^3$/s)',
             color='blue')
    plt.plot(months, runoff_timeseries_russia, label='Russia (Mean: '+f'{np.mean(runoff_timeseries_russia):.0f})' + ' m$^3$/s)',
             color='red')
    plt.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
    plt.ylabel('m$^3$/s')
    plt.title('Climatological Runoff from Fekete et al 2002')

    ######################################################################################
    # Timeseries from Blaskey et al 2023

    ax = plt.subplot(gs[1, :])
    plt.title('Alaska Runoff from Blaskey et al 2023')
    runoff_blaskey_climatology = np.mean(runoff_blaskey[:, :365], axis=0)

    vertical_line_locations = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])
    for vline in vertical_line_locations:
        plt.axvline(x=vline, color='k', linestyle='--', alpha=0.5, linewidth=0.5)

    for i in range(runoff_blaskey.shape[0]):
        if i==0:
            plt.plot(runoff_blaskey[i, :], color='silver', alpha=0.5, label='Individual Years')
        else:
            plt.plot(runoff_blaskey[i, :], color='silver', alpha=0.5)
    plt.ylabel('m$^3$/s')

    plt.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)

    plt.plot(runoff_blaskey_climatology,
             label='Climatology (Mean: ' + f'{np.mean(runoff_blaskey_climatology):.0f})' + ' m$^3$/s)',
             color='blue')
    plt.legend()
    xticks = np.diff(vertical_line_locations) / 2 + vertical_line_locations[:-1]
    xtick_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(xticks, xtick_labels)

    # # plot the runoff in 2023
    # ax = plt.subplot(gs[2, :])
    # plt.title('Estimated Runoff in 2023')
    # runoff_timeseries_2023 = np.sum(runoff_2023, axis=(1, 2))
    # runoff_timeseries_2023_alaska = np.sum(runoff_2023[:, :, 580:], axis=(1, 2))
    # runoff_timeseries_2023_russia = np.sum(runoff_2023[:, :, :580], axis=(1, 2))
    # plt.plot(runoff_timeseries_2023, label='Total (Mean: '+f'{np.mean(runoff_timeseries_2023):.0f})' + ' m$^3$/s)',
    #          color='orange')
    # plt.plot(runoff_timeseries_2023_alaska, label='Alaska (Mean: '+f'{np.mean(runoff_timeseries_2023_alaska):.0f})' + ' m$^3$/s)',
    #          color='blue')
    # plt.plot(runoff_timeseries_2023_russia, label='Russia (Mean: '+f'{np.mean(runoff_timeseries_2023_russia):.0f})' + ' m$^3$/s)',
    #          color='red')
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
    #
    #
    # # plot the runoff in 2024
    # ax = plt.subplot(gs[3, :])
    # plt.title('Estimated Runoff in 2024')
    # runoff_timeseries_2024 = np.sum(runoff_2024, axis=(1, 2))
    # runoff_timeseries_2024_alaska = np.sum(runoff_2024[:, :, 580:], axis=(1, 2))
    # runoff_timeseries_2024_russia = np.sum(runoff_2024[:, :, :580], axis=(1, 2))
    # plt.plot(runoff_timeseries_2024, label='Total (Mean: '+f'{np.mean(runoff_timeseries_2024):.0f})' + ' m$^3$/s)',
    #          color='orange')
    # plt.plot(runoff_timeseries_2024_alaska, label='Alaska (Mean: '+f'{np.mean(runoff_timeseries_2024_alaska):.0f})' + ' m$^3$/s)',
    # color='blue')
    # plt.plot(runoff_timeseries_2024_russia, label='Russia (Mean: '+f'{np.mean(runoff_timeseries_2024_russia):.0f})' + ' m$^3$/s)',
    # color='red')
    # plt.xlabel('Day of the Year')
    # plt.legend()
    # # print(runoff_timeseries_2024)

    plt.savefig(os.path.join(project_folder, 'Figures', 'Model', 'Chukchi_Sea_Runoff.png'),dpi=300)
    plt.close(fig)

testing = False

project_folder = '/Users/mike/Documents/Research/Projects/Chukchi Sea'

runoff_fekete_climatology = read_Fekete_runoff_data(project_folder)
runoff_blaskey = read_Blaskey_runoff_data(project_folder)

# runoff_2023, runoff_2024 = read_reconstructed_runoff_data(project_folder)
#
# XC, YC, Depth, rA = read_model_grid(project_folder)
# for month in range(12):
#     runoff_fekete_climatology[month, :, :] *= rA  # Convert from m/s to m3/s
# for day in range(366):
#     if day<365:
#         runoff_2023[day, :, :] *= rA  # Convert from m/s to m3/s
#     runoff_2024[day, :, :] *= rA  # Convert from m/s to m3/s

plot_summer_runoff(project_folder, runoff_fekete_climatology, runoff_blaskey)#, runoff_2023, runoff_2024)






