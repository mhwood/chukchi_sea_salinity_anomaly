

import os
import argparse
import sys
import numpy as np
import netCDF4 as nc4
from pyproj import Transformer
from scipy.interpolate import griddata
import matplotlib.pyplot as plt



def read_extent_from_model_grid_nc(config_dir, model_name):
    ds = nc4.Dataset(os.path.join(config_dir, 'nc_grids', model_name + '_grid.nc'))
    Lon = ds.variables['XC'][:, :]
    Lat = ds.variables['YC'][:, :]
    Depth = ds.variables['Depth'][:, :]
    ds.close()

    return (Lon, Lat, Depth)

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
    elif str(inputCRS)[:2] == '34' and outputCRS == 32602:
        x2, y2 = transformer.transform(polygon_array[:, x_column], polygon_array[:, y_column])
        x2 = np.array(x2)
        y2 = np.array(y2)
    else:
        raise ValueError('Reprojection with this epsg is not safe - no test for validity has been implemented')

    output_polygon=np.copy(polygon_array)
    output_polygon[:,x_column] = x2
    output_polygon[:,y_column] = y2
    return output_polygon

def read_seaice_anomaly(seaice_dir, month):

    version = 5
    if month in [8]:
        days_in_month=31
    if month in [9]:
        days_in_month=30

    sum_grid = np.zeros((448, 304))
    count_grid = np.zeros((448, 304))
    sum_grid_2024 = np.zeros((448, 304))
    count_grid_2024 = np.zeros((448, 304))
    count = 0
    for year in range(2015,2025):
        sat_code = 'F17'
        if version==5:
            for day in range(1, days_in_month + 1):
                file_name = 'sic_psn25_' + str(year) + str(month).zfill(2) + str(day).zfill(2) + '_'+sat_code+'_v05r00.nc'
                # if os.path.exists(os.path.join(SSS_dir, file_name)):
                ds = nc4.Dataset(os.path.join(seaice_dir, 'v5', file_name))
                seaice = np.array(ds.variables['cdr_seaice_conc'][:, :, :])
                seaice = seaice[0,:,:]
                x = ds.variables['x'][:]
                y = ds.variables['y'][:]
                ds.close()
                valid_indices = seaice<=1
                sum_grid[valid_indices] += seaice[valid_indices]
                count_grid[valid_indices] += 1
                if year == 2024:
                    sum_grid_2024[valid_indices] += seaice[valid_indices]
                    count_grid_2024[valid_indices] += 1

    seaice_mean = np.full((448, 304), np.nan)
    seaice_mean[count_grid > 0] = sum_grid[count_grid > 0] / count_grid[count_grid > 0]

    seaice_mean_2024 = np.full((448, 304), np.nan)
    seaice_mean_2024[count_grid_2024 > 0] = sum_grid_2024[count_grid_2024 > 0] / count_grid_2024[count_grid_2024 > 0]

    valid_indices_2024 = seaice_mean_2024 >= 0
    SSS_anomaly = np.full((448, 304), np.nan)
    SSS_anomaly[valid_indices_2024] = seaice_mean_2024[valid_indices_2024] - seaice_mean[valid_indices_2024]

    return x, y, SSS_anomaly

def interpolate_seaice_anomaly_to_grid(x, y, seaice_anomaly, X, Y):

    X_3411, Y_3411 = np.meshgrid(x, y)
    # plt.pcolormesh(X_3411, Y_3411, seaice)
    # plt.show()

    points = np.column_stack([X_3411.ravel(), Y_3411.ravel()])
    points = reproject_points(points, 3411, 32602)
    X_32602_seaice = np.reshape(points[:, 0], np.shape(X_3411))
    Y_32602_seaice = np.reshape(points[:, 1], np.shape(Y_3411))

    points_nonnan = np.column_stack([X_32602_seaice.ravel(), Y_32602_seaice.ravel()])
    seaice_nonnan = np.ravel(seaice_anomaly)
    non_zero_locations = seaice_nonnan < 2
    # points_nonnan = points_nonnan[non_zero_locations, :]
    # seaice_nonnan = seaice_nonnan[non_zero_locations]
    print(np.shape(points_nonnan), np.shape(seaice_nonnan))

    seaice_nearest = griddata(points_nonnan, seaice_nonnan, (X, Y), method='linear')

    return(seaice_nearest)

def write_seaice_anomaly_to_nc(project_dir, month, X, Y, seaice_anomaly):
    if month==8:
        month_name = 'August'
    if month==9:
        month_name = 'September'
    output_file = os.path.join(project_dir,'Data', 'Observations', 'Sea_ice_Anomaly_2025_'+month_name+'.nc')
    with nc4.Dataset(output_file, 'w', format='NETCDF4') as ds:
        ds.createDimension('x', X.shape[1])
        ds.createDimension('y', Y.shape[0])

        x_var = ds.createVariable('Longitude', 'f4', ('y', 'x'))
        y_var = ds.createVariable('Latitude', 'f4', ('y', 'x'))
        SI_var = ds.createVariable('seaice_anomaly', 'f8', ('y', 'x'))

        x_var[:] = X
        y_var[:] = Y
        SI_var[:] = seaice_anomaly


home_dir = os.path.expanduser('~')

project_dir = home_dir+'/Documents/Research/Projects/Chukchi_Sea'

config_dir = home_dir+'/Documents/Research/Projects/Ocean_Modelling/Projects/' \
             'Downscale_Greenland/MITgcm/configurations/downscale_greenland/'

seaice_dir = '/Volumes/CoOL/Data_Repository/Arctic/Sea_Ice/daily'

model_name = 'Chukchi_Sea'
month = 9

Lon, Lat, Depth = read_extent_from_model_grid_nc(config_dir, model_name)

min_lon = np.min(Lon)
max_lon = np.max(Lon)
min_lat = np.min(Lat)
max_lat = np.max(Lat)

resolution = 1000

print(' - Creating the output grids')
points = np.column_stack([Lon.ravel(), Lat.ravel()])
points = reproject_points(points, inputCRS=4326, outputCRS=32602)
X_plot = np.reshape(points[:, 0], Lon.shape)
Y_plot = np.reshape(points[:, 1], Lat.shape)
x = np.arange(np.min(points[:, 0]) - resolution, np.max(points[:, 0]) + 2 * resolution, resolution)
y = np.arange(np.min(points[:, 1]) - resolution, np.max(points[:, 1]) + 2 * resolution, resolution)
X, Y = np.meshgrid(x, y)
epsg = 32602


x_seaice, y_seaice, seaice_anomaly = read_seaice_anomaly(seaice_dir, month=month)

# plt.pcolormesh(x_seaice, y_seaice, seaice_anomaly, cmap='RdBu_r', shading='auto', vmin=-0.5, vmax=0.5)
# plt.colorbar()
# plt.show()

seaice_anomaly = interpolate_seaice_anomaly_to_grid(x_seaice, y_seaice, seaice_anomaly, X, Y)

plt.pcolormesh(X, Y, seaice_anomaly, cmap='RdBu_r', shading='auto', vmin=-0.5, vmax=0.5)
plt.colorbar()
plt.show()

write_seaice_anomaly_to_nc(project_dir, month, X, Y, seaice_anomaly)




