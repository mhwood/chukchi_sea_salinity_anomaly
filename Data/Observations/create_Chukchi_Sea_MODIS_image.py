import os
import argparse
import sys
import numpy as np
import netCDF4 as nc4
from pyproj import Transformer
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
# from osgeo import gdal
# from osgeo import osr


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

def read_RGB_bands_from_nc(nc_file_path):
    ds = nc4.Dataset(nc_file_path)
    Lon = ds.variables['Longitude'][:, :]
    Lat = ds.variables['Latitude'][:, :]
    band_1 = ds.variables['sur_refl_b01'][:, :, :]
    band_1 = band_1[0, :, :]
    band_3 = ds.variables['sur_refl_b03'][:, :, :]
    band_3 = band_3[0, :, :]
    band_4 = ds.variables['sur_refl_b04'][:, :, :]
    band_4 = band_4[0, :, :]
    ds.close()

    return (Lon, Lat, band_1, band_3, band_4)

def read_MODIS_points_to_domain(modis_path, min_lon, max_lon, min_lat, max_lat, reproject_to_polar, epsg):

    # https://opendap.cr.usgs.gov/opendap/hyrax/MYD09A1.061/h18v01.ncml.dmr.html

    file_names = ['h09v02.ncml.nc4','h10v02.ncml.nc4','h11v02.ncml.nc4','h11v01.ncml.nc4',
                  'h25v02.ncml.nc4','h26v02.ncml.nc4','h12v01.ncml.nc4','h12v02.ncml.nc4',
                  'h24v02.ncml.nc4','h23v01.ncml.nc4','h23v02.ncml.nc4','h13v01.ncml.nc4']

    resolution_buffer = 0.1

    points_started = False

    for file_name in file_names:
        print('   - Reading ' + file_name)
        nc_file_path = modis_path + '/' + file_name

        Lon, Lat, band_1, band_3, band_4 = read_RGB_bands_from_nc(nc_file_path)

        # wrap the lon
        Lon[Lon<0] += 360  # wrap longitudes to 0-360 range

        skip = 5
        Lon = Lon[::skip, ::skip]  # downsample to reduce size
        Lat = Lat[::skip, ::skip]  # downsample to reduce size
        band_1 = band_1[::skip, ::skip]  # downsample to reduce size
        band_3 = band_3[::skip, ::skip]  # downsample to reduce size
        band_4 = band_4[::skip, ::skip]  # downsample to reduce size

        points = np.column_stack([np.ravel(Lon), np.ravel(Lat)])
        print(np.min(Lon), np.max(Lon), np.min(Lat), np.max(Lat), 'Lon/Lat range')

        values_1 = np.reshape(band_1, (np.size(band_1), 1))  # red
        values_3 = np.reshape(band_3, (np.size(band_3), 1))  # green
        values_4 = np.reshape(band_4, (np.size(band_4), 1))  # blue

        indices_lon = np.logical_and(points[:, 0] >= min_lon - resolution_buffer,
                                     points[:, 0] <= max_lon + resolution_buffer)
        indices_lat = np.logical_and(points[:, 1] >= min_lat - resolution_buffer,
                                     points[:, 1] <= max_lat + resolution_buffer)
        indices = np.logical_and(indices_lon, indices_lat)

        if np.any(indices):

            points_subset = points[indices, :]
            band_1_subset = values_1[indices]
            band_3_subset = values_3[indices]
            band_4_subset = values_4[indices]

            if not points_started:
                points_started = True
                all_points = points_subset
                all_band_1_points = band_1_subset
                all_band_3_points = band_3_subset
                all_band_4_points = band_4_subset
            else:
                all_points = np.vstack([all_points, points_subset])
                all_band_1_points = np.vstack([all_band_1_points, band_1_subset])
                all_band_3_points = np.vstack([all_band_3_points, band_3_subset])
                all_band_4_points = np.vstack([all_band_4_points, band_4_subset])

    if reproject_to_polar:
        all_points = reproject_points(all_points, inputCRS=4326, outputCRS=epsg)

    #filter out all points that are less than -0.1
    band_1_indices = all_band_1_points > -0.1
    band_3_indices = all_band_3_points > -0.1
    band_4_indices = all_band_4_points > -0.1
    non_zero_indices = np.logical_and(band_1_indices, band_3_indices, band_4_indices).ravel()
    all_points = all_points[non_zero_indices, :]
    all_band_1_points = all_band_1_points[non_zero_indices]
    all_band_3_points = all_band_3_points[non_zero_indices]
    all_band_4_points = all_band_4_points[non_zero_indices]

    return (all_points, all_band_1_points, all_band_3_points, all_band_4_points)

def interpolate_points_to_grid(points, X, Y, band_1_points, band_3_points, band_4_points):
    print('    - Interpolating band 1')
    band_1_grid = griddata(points, band_1_points.ravel(), (X, Y), method='nearest')
    print('    - Interpolating band 3')
    band_3_grid = griddata(points, band_3_points.ravel(), (X, Y), method='nearest')
    print('    - Interpolating band 4')
    band_4_grid = griddata(points, band_4_points.ravel(), (X, Y), method='nearest')

    return (band_1_grid, band_3_grid, band_4_grid)

def write_data_to_tif(output_file, epsg, x, y, band_1_grid, band_3_grid, band_4_grid):
    geotransform = (np.min(x), x[1] - x[0], 0, np.max(y), 0, y[0] - y[1])

    output_raster = gdal.GetDriverByName('GTiff').Create(output_file, len(x), len(y), 3,
                                                         gdal.GDT_Float32)  # Open the file
    output_raster.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)

    output_raster.SetProjection(srs.ExportToWkt())
    output_raster.GetRasterBand(1).WriteArray(np.flipud(band_3_grid))  # Writes my array to the raster
    output_raster.GetRasterBand(2).WriteArray(np.flipud(band_4_grid))  # Writes my array to the raster
    output_raster.GetRasterBand(3).WriteArray(np.flipud(band_1_grid))  # Writes my array to the raster

    output_raster.FlushCache()

def write_data_to_nc(output_file, x, y, band_1_grid, band_3_grid, band_4_grid):
    ds = nc4.Dataset(output_file, 'w', format='NETCDF4')

    # Create dimensions
    ds.createDimension('x', len(x))
    ds.createDimension('y', len(y))

    # Create variables
    lon_var = ds.createVariable('x', 'f4', ('x',))
    lat_var = ds.createVariable('y', 'f4', ('y',))
    band_1_var = ds.createVariable('band_1', 'f4', ('y', 'x'))
    band_3_var = ds.createVariable('band_3', 'f4', ('y', 'x'))
    band_4_var = ds.createVariable('band_4', 'f4', ('y', 'x'))

    # Assign data to variables
    lon_var[:] = x
    lat_var[:] = y
    band_1_var[:] = band_1_grid
    band_3_var[:] = band_3_grid
    band_4_var[:] = band_4_grid

    ds.close()

def create_MODIS_file(project_dir, config_dir, modis_path, model_name):
    buffer = 0.1

    reproject_to_polar = True
    resolution = 1000#250

    # reproject_to_polar = False
    # resolution = 0.1

    # step 1: read in the geometry
    print(' - Reading in the model geometry')
    Lon, Lat, Depth = read_extent_from_model_grid_nc(config_dir, model_name)

    min_lon = np.min(Lon)
    max_lon = np.max(Lon)
    min_lat = np.min(Lat)
    max_lat = np.max(Lat)

    # step 2: reproject to polar coorindates
    print(' - Creating the output grids')
    if reproject_to_polar:
        points = np.column_stack([Lon.ravel(), Lat.ravel()])
        points = reproject_points(points, inputCRS=4326, outputCRS=32602)
        X_plot = np.reshape(points[:, 0], Lon.shape)
        Y_plot = np.reshape(points[:, 1], Lat.shape)
        x = np.arange(np.min(points[:, 0]) - resolution, np.max(points[:, 0]) + 2 * resolution, resolution)
        y = np.arange(np.min(points[:, 1]) - resolution, np.max(points[:, 1]) + 2 * resolution, resolution)
        X, Y = np.meshgrid(x, y)
        epsg = 32602
    else:
        x = np.arange(min_lon - resolution, max_lon + 2 * resolution, resolution)
        y = np.arange(min_lat - resolution, max_lat + 2 * resolution, resolution)
        X, Y = np.meshgrid(x, y)
        epsg = 4326

    # step 3: read in the modis points
    print(' - Reading in the MODIS data')
    points, band_1_points, band_3_points, band_4_points = \
        read_MODIS_points_to_domain(modis_path, min_lon, max_lon, min_lat, max_lat, reproject_to_polar, epsg)

    # nonzero_band_1_indices = band_1_points > 0
    # nonzero_band_3_indices = band_3_points > 0
    # nonzero_band_4_indices = band_4_points > 0
    # non_zero_indices = np.logical_and(nonzero_band_1_indices, nonzero_band_3_indices, nonzero_band_4_indices).ravel()
    # print(np.shape(points), 'Points shape')
    # print(np.shape(band_1_points), 'Band 1 shape')
    # print(np.shape(band_3_points), 'Band 3 shape')
    # print(np.shape(band_4_points), 'Band 4 shape')
    # print(np.shape(non_zero_indices), 'Non-zero indices shape')
    # points = points[non_zero_indices, :]
    # band_1_points = band_1_points[non_zero_indices]
    # band_3_points = band_3_points[non_zero_indices]
    # band_4_points = band_4_points[non_zero_indices]

    # print(np.min(band_1_points), np.max(band_1_points), 'Band 1 range')

    # print(np.isnan(band_1_points).sum(), 'NaN values in band 1')

    # plt.plot(points[:, 0], points[:, 1], 'k.', markersize=1)
    # plt.plot(X[:,-1],Y[:,-1], 'g-')
    # plt.plot(X[0,:],Y[0,:], 'g-')
    # plt.plot(X[:,0],Y[:,0], 'g-')
    # plt.plot(X[-1, :],Y[-1, :], 'g-')
    # plt.contour(Lon, Lat, Depth, levels=10, colors='silver', linewidths=0.5, linestyles='solid', alpha=0.5)
    # plt.show()


    # step 4: interpolate the modis points onto the grid
    print(' - Interpolating the points onto the domain')
    band_1_grid, band_3_grid, band_4_grid = \
        interpolate_points_to_grid(points, X, Y, band_1_points, band_3_points, band_4_points)

    # C = plt.imshow(band_1_grid, origin='lower')
    # plt.colorbar(C)
    # plt.show()

    # step 5: output the files to tif
    # print(' - Outputting the bands to tif')
    # output_file = os.path.join(config_dir, 'L2', model_name, 'plots',
    #                            model_name + '_MODIS_20220720_' + str(epsg) + '.tif')
    # write_data_to_tif(output_file, epsg, x, y, band_1_grid, band_3_grid, band_4_grid)

    print(' - Outputting the bands to nc')
    output_file = os.path.join(project_dir, 'Imagery',
                               model_name + '_MODIS_20220720_' + str(epsg) + '.nc')
    write_data_to_nc(output_file, x, y, band_1_grid, band_3_grid, band_4_grid)


project_dir = '/Users/mhwood/Documents/Research/Projects/Chukchi_Sea/Data'

config_dir = '/Users/mhwood/Documents/Research/Projects/Ocean_Modelling/Projects/' \
             'Downscale_Greenland/MITgcm/configurations/downscale_greenland/'

modis_path = '/Users/mhwood/Documents/Research/Data Repository/Greenland/MODIS/'

model_name = 'Chukchi_Sea'

create_MODIS_file(project_dir, config_dir, modis_path,model_name)





