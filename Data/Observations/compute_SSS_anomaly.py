

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
    else:
        raise ValueError('Reprojection with this epsg is not safe - no test for validity has been implemented')

    output_polygon=np.copy(polygon_array)
    output_polygon[:,x_column] = x2
    output_polygon[:,y_column] = y2
    return output_polygon

def read_global_SS_anomaly(SSS_dir, month):

    version = 6

    sum_grid = np.zeros((720, 1440))
    count_grid = np.zeros((720, 1440))
    count = 0
    for year in range(2015,2026):
        if version==5:
            file_name = 'SMAP_L3_SSS_' + str(year) + str(month).zfill(2) + '_MONTHLY_V5.0.nc'
            # if os.path.exists(os.path.join(SSS_dir, file_name)):
            ds = nc4.Dataset(os.path.join(SSS_dir, 'v5', file_name))
            SSS = np.array(ds.variables['smap_sss'][:, :])
            longitude = ds.variables['longitude'][:]
            latitude = ds.variables['latitude'][:]
            ds.close()
        if version==6:
            file_name = 'RSS_smap_SSS_L3_monthly_' + str(year) +'_'+ str(month).zfill(2) + '_FNL_v06.0.nc'
            # if os.path.exists(os.path.join(SSS_dir, file_name)):
            ds = nc4.Dataset(os.path.join(SSS_dir, 'v6', file_name))
            SSS = np.array(ds.variables['sss_smap'][:, :])
            longitude = ds.variables['lon'][:]
            latitude = ds.variables['lat'][:]
            ds.close()
        valid_indices = SSS>0
        sum_grid[valid_indices] += SSS[valid_indices]
        count_grid[valid_indices] += 1
        if year == 2024:
            SSS_2024 = SSS

    SSS_mean = np.full((720, 1440), np.nan)
    SSS_mean[count_grid > 0] = sum_grid[count_grid > 0] / count_grid[count_grid > 0]

    valid_indices_2024 = SSS_2024 > 0
    SSS_anomaly = np.full((720, 1440), np.nan)
    SSS_anomaly[valid_indices_2024] = SSS_2024[valid_indices_2024] - SSS_mean[valid_indices_2024]

    return longitude, latitude, SSS_anomaly

def write_SSS_anomaly_to_nc(project_dir, month, X, Y, SSS_anomaly):
    if month==8:
        month_name = 'August'
    if month==9:
        month_name = 'September'
    output_file = os.path.join(project_dir,'Data', 'Observations', 'SMAP_SSS_Anomaly_2025_'+month_name+'.nc')
    with nc4.Dataset(output_file, 'w', format='NETCDF4') as ds:
        ds.createDimension('x', X.shape[1])
        ds.createDimension('y', Y.shape[0])

        x_var = ds.createVariable('Longitude', 'f4', ('y', 'x'))
        y_var = ds.createVariable('Latitude', 'f4', ('y', 'x'))
        SSS_var = ds.createVariable('SSS_anomaly', 'f8', ('y', 'x'))

        x_var[:] = X
        y_var[:] = Y
        SSS_var[:] = SSS_anomaly


home_dir = os.path.expanduser('~')

project_dir = home_dir+'/Documents/Research/Projects/Chukchi_Sea'

config_dir = home_dir+'/Documents/Research/Projects/Ocean_Modelling/Projects/' \
             'Downscale_Greenland/MITgcm/configurations/downscale_greenland/'

SSS_dir = '/Volumes/CoOL/Data_Repository/Global/Sea Surface Salinity/Monthly'

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


longitude, latitude, SSS_anomaly = read_global_SS_anomaly(SSS_dir, month=month)
Longitude, Latitude = np.meshgrid(longitude, latitude)

valid_lat = (latitude >= min_lat-2) & (latitude <= max_lat+2)
Longitude = Longitude[valid_lat, :]
Latitude = Latitude[valid_lat, :]
SSS_anomaly = SSS_anomaly[valid_lat, :]

SSS_points = np.column_stack([Longitude.ravel(), Latitude.ravel()])
SSS_points = reproject_points(SSS_points, inputCRS=4326, outputCRS=epsg)

SSS_gridded = griddata(SSS_points, SSS_anomaly.ravel(), (X, Y), method='linear')

# plt.pcolormesh(X, Y, SSS_gridded, cmap='RdBu_r', shading='auto',vmin=-5,vmax=5)
# plt.colorbar(label='SSS Anomaly (psu)')
# plt.show()

write_SSS_anomaly_to_nc(project_dir, month, X, Y, SSS_gridded)




