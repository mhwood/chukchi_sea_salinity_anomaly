

import os
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc4
from pyproj import Transformer
import cmocean.cm as cm
from matplotlib.patches import Rectangle
from datetime import timedelta, datetime

from matplotlib.gridspec import GridSpec

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def read_MODIS_imagery(project_folder):

    ds = nc4.Dataset(os.path.join(project_folder, 'Data','Imagery', 'Chukchi_Sea_MODIS_20220720_32602.nc'))
    x = ds.variables['x'][:]
    y = ds.variables['y'][:]
    band_1 = ds.variables['band_1'][:,:]
    band_3 = ds.variables['band_3'][:,:]
    band_4 = ds.variables['band_4'][:,:]
    ds.close()

    # flipud all the bands
    band_1 = np.flipud(band_1)
    band_3 = np.flipud(band_3)
    band_4 = np.flipud(band_4)

    # normalize the bands
    band_1 = band_1/1.6
    band_3 = band_3/1.6
    band_4 = band_4/1.6

    band_1[band_1 < 0] = 0
    band_3[band_3 < 0] = 0
    band_4[band_4 < 0] = 0

    img = np.concatenate((band_1[:,:,np.newaxis], band_4[:,:,np.newaxis], band_3[:,:,np.newaxis]), axis=2)

    # brightness correction
    img = img / np.max(img)
    img = np.clip(img, 0, 1)


    #increase brightness
    img = img * 4
    img[img > 1] = 1

    img = (img * 255).astype(np.uint8)

    return(x,y, img)

def read_model_grid(project_folder):
    ds = nc4.Dataset(os.path.join(project_folder,'Data', 'Model', 'Chukchi_Sea_grid.nc'))
    XC = ds.variables['XC'][:,:]
    YC = ds.variables['YC'][:,:]
    Depth = ds.variables['Depth'][:,:]
    ds.close()
    return(XC, YC, Depth)

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

def read_model_field_from_mds(project_folder, var_name, iter_number):

    depth_level = 5

    if var_name=='SIheff':
        subset = 'SI_day_snap'
        file_name = subset + '.' + str(iter_number).zfill(10) + '.data'
        n_levels = 5
        lev = 1
    elif var_name=='Theta':
        subset = 'state_3D_day_mean'
        file_name = subset + '.' + str(iter_number).zfill(10) + '.data'
        n_levels = 90*2
        lev = depth_level
    elif var_name=='Salt':
        subset = 'state_3D_day_mean'
        file_name = subset + '.' + str(iter_number).zfill(10) + '.data'
        n_levels = 90*2
        lev = depth_level + 90

    data = np.fromfile(os.path.join(project_folder,'run','diags',subset,file_name), dtype='>f4')
    data = data.reshape((n_levels, 720, 960))
    data = data[lev,:,:]

    return data

def var_name_to_subset_and_filename(var_name, date_str):
    if var_name in ['Theta','Salt']:
        subset = 'daily_mean'
        file_name = var_name + '_' + date_str[:6] + '.nc'
    elif var_name in ['SIheff','SIarea']:
        subset = 'daily_snapshot'
        file_name = var_name + '_' + date_str[:6] + '.nc'
    return subset, file_name

def read_model_field_from_nc(config_dir, results_dir, var_name, year, month):
    depth_level = 5

    if var_name in ['Theta','Salt']:
        subset = 'daily_mean'
        file_name = var_name + '_' + f'{year}{month:02d}' + '.nc'
        lev = depth_level
    elif var_name in ['SIheff','SIarea']:
        subset = 'daily_snapshot'
        file_name = var_name + '_' + f'{year}{month:02d}' + '.nc'
        lev = 0

    ds = nc4.Dataset(os.path.join(config_dir, results_dir, subset, var_name, file_name))
    if var_name in ['Theta', 'Salt']:
        data = ds.variables[var_name][:,lev,:,:]
    elif var_name in ['SIheff','SIarea']:
        data = ds.variables[var_name][:,:,:]
    ds.close()

    return data

def plot_anomaly_panel(project_folder, experiment, year, var_name, data, month, x, y, img, Depth):

    if month==8:
        month_name = 'August'
    elif month==6:
        month_name = 'June'
    elif month==7:
        month_name = 'July'
    elif month==9:
        month_name = 'September'
    elif month==10:
        month_name = 'October'

    file_name = var_name+'_'+str(year)+'_'+month_name+'_anomaly_'+experiment+'_vs_control.png'
    output_folder = os.path.join(project_folder, 'Figures','Anomaly Maps vs Control')

    plot_metadata = {'SIheff': {'units': 'm', 'long_name': 'Sea Ice Thickness', 'cmap': cm.balance, 'vmin': -0.5, 'vmax': 0.5},
                     'SIarea': {'units': 'm$^2$/m$^2$', 'long_name': 'Sea Ice Concentration', 'cmap': cm.balance, 'vmin': -0.5, 'vmax': 0.5},
                     'Theta': {'units': 'C', 'long_name': 'Sea Surface Temperature', 'cmap': cm.balance, 'vmin': -3,
                               'vmax': 3},
                     'Salt': {'units': 'psu', 'long_name': 'Salinity', 'cmap': cm.balance, 'vmin': -1, 'vmax': 1},
                     'UVEL': {'units': 'm/s', 'long_name': 'Eastward Velocity', 'cmap': 'viridis', 'vmin': -0.5,
                              'vmax': 0.5},
                     'VVEL': {'units': 'm/s', 'long_name': 'Northward Velocity', 'cmap': 'viridis', 'vmin': -0.5,
                              'vmax': 0.5},
                     'Speed': {'units': 'm/s', 'long_name': 'Speed', 'cmap': 'viridis', 'vmin': 0, 'vmax': 1}}


    fig = plt.figure(figsize=(10, 8))
    plt.style.use('dark_background')

    gs = GridSpec(17, 11, figure=fig, left = 0.02, right = 0.99, top = 0.91, bottom = 0.12)

    ax = fig.add_subplot(gs[:, :])

    plot_grid = np.ma.masked_where(Depth == 0, data)

    # add a white Rectangle
    rect = Rectangle((x.min(), y.min()), x.max() - x.min(), y.max() - y.min(), linewidth=1, edgecolor='w',
                     facecolor='white', zorder=0)
    plt.gca().add_patch(rect)

    plt.imshow(img, extent=(x.min(), x.max(), y.min(), y.max()), alpha=0.8, zorder=2)
    C = plt.imshow(plot_grid, extent=(X.min(), X.max(), Y.min(), Y.max()), cmap=plot_metadata[var_name]['cmap'],
                   origin='lower',
                   vmin=plot_metadata[var_name]['vmin'], vmax=plot_metadata[var_name]['vmax'], zorder=3)
    plt.colorbar(C, label=plot_metadata[var_name]['units'], orientation='vertical', pad=0.02, aspect=40,
                 shrink=0.8, ticks=np.linspace(plot_metadata[var_name]['vmin'], plot_metadata[var_name]['vmax'], 6))

    # add a white contour on the coastlines
    plt.contour(X, Y, Depth, levels=[0], colors='w', linewidths=0.5, zorder=4)

    # if var_name=='Speed':
    #     # add a quiver
    #     skip = 20
    #     U = U[::skip, ::skip]
    #     V = V[::skip, ::skip]
    #     U[U==0] = np.nan
    #     V[V==0] = np.nan
    #     plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], U, V, color='w', scale=10)

    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])

    if experiment=='daily_hydrology':
        experiment_long_name = 'Daily Hydrology & Model-Evolving Sea Ice'

    if experiment=='prescribed_seaice':
        experiment_long_name = 'Daily Hydrology & Nudged Sea Ice'
    experiment_long_name_control = 'Climatological Hydrology & Model-Evolving Sea Ice'

    if 'SI' in var_name:
        plt.title(plot_metadata[var_name]['long_name'] + ' Anomaly'+\
                  '\nComparison: ' + experiment_long_name +' vs \n' + experiment_long_name_control,
                  fontsize=12)
    else:
        plt.title(plot_metadata[var_name]['long_name'] + ' Anomaly (5m depth)'+\
                  '\nComparison: ' + experiment_long_name +' vs \n' + experiment_long_name_control,
                  fontsize=12)

    # add labels for Alaska and Russia with semitransparent bounding boxes
    ax.text(x.min() + 0.85 * (x.max() - x.min()), y.min() + 0.65 * (y.max() - y.min()), 'Alaska',fontsize=12,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'),
            ha='right', va='center', zorder=5)
    ax.text(x.min() + 0.2 * (x.max() - x.min()), y.min() + 0.6 * (y.max() - y.min()), 'Russia', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'),
            ha='left', va='top', zorder=5)

    # set the extent to remove 3% from edges
    plt.xlim(x.min() + 0.03 * (x.max() - x.min()), x.max() - 0.03 * (x.max() - x.min()))
    plt.ylim(y.min() + 0.03 * (y.max() - y.min()), y.max() - 0.03 * (y.max() - y.min()))

    plt.savefig(
        os.path.join(output_folder, file_name),
        dpi=300)
    plt.close(fig)




config_dir = '/Volumes/chukchi_sea/Research/Ocean_Modeling/Projects/Chukchi_Sea/'
# config_dir = '/Volumes/ikinngut/Research/Model Results/Chukchi Sea'

project_folder = '/Users/mhwood/Documents/Research/Projects/Chukchi_Sea'

year = 2024

for experiment in ['prescribed_seaice','daily_hydrology']:
    print('Processing results for the ' + experiment+' experiment')
    for var_name in ['SIarea','Theta','Salt',]:
        print('  Working on variable ' + var_name)
        for month in [6,7,8,9,10]:
            print('    Working on month ' + str(month))

            x,y,img = read_MODIS_imagery(project_folder)

            XC, YC, Depth = read_model_grid(project_folder)

            # reproject the model grid to 32602
            points = reproject_points(np.column_stack((XC.flatten(), YC.flatten())), 4326, 32602)
            X = points[:, 0].reshape(XC.shape)
            Y = points[:, 1].reshape(YC.shape)

            subset, file_name = var_name_to_subset_and_filename(var_name, f'2024{month:02d}')
            print('      Reading data from ' + file_name)

            results_dir = 'results_'+experiment
            results_dir_control = 'results_control'

            data = read_model_field_from_nc(config_dir, results_dir, var_name, year, month)
            data_control = read_model_field_from_nc(config_dir, results_dir_control, var_name, year, month)

            data_difference = data - data_control
            data_difference = np.mean(data_difference, axis=0)

            plot_anomaly_panel(project_folder, experiment, year, var_name, data_difference, month, x, y, img, Depth)

