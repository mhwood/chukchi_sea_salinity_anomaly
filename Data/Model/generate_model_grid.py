
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import netCDF4 as nc4
from pyproj import Transformer
from scipy.interpolate import griddata

def read_coarse_bathymetry_from_nc():
    file_name = '/Users/mhwood/Documents/Research/Projects/Chukchi_Sea/Data/global_bathymetry.nc'
    ds = nc4.Dataset(file_name)
    lat = ds.variables['latitude'][:]
    lon = ds.variables['longitude'][:]
    depth = ds.variables['Depth'][:,:]
    ds.close()

    depth = np.array(depth)

    # find index of negative

    # switch half of the lon grid and depth grid to the other side to account for wrappign longitude
    bathy = np.hstack((depth[:, int(len(lon)/2):], depth[:, :int(len(lon)/2)]))
    lon = np.concatenate((lon[int(len(lon)/2):], lon[:int(len(lon)/2)]))
    lon[lon<0] += 360  # ensure all longitudes are positive

    # subset to the locations around Greenland
    min_lat = 58
    max_lat = 76
    min_lon = 155
    max_lon = 220
    lat_indices = (lat >= min_lat) & (lat <= max_lat)
    lon_indices = (lon >= min_lon) & (lon <= max_lon)
    lat = lat[lat_indices]
    lon = lon[lon_indices]
    depth = depth[lat_indices,:]
    depth = depth[:,lon_indices]
    bathy = bathy[lat_indices,:]
    bathy = bathy[:,lon_indices]

    # plot the bathymetry
    # plt.subplot(1, 2, 1)
    # plt.pcolormesh(bathy)
    # plt.subplot(1, 2, 2)
    # plt.pcolormesh(depth)
    # plt.show()
    return lon, lat, bathy

def reproject_polygon(polygon_array,inputCRS,outputCRS,x_column=0,y_column=1,run_test = True):

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

def create_grid(sNx, sNy, resolution):
    # Define the bounding box
    min_y = 6800386
    min_x = -533729
    max_y = 8147621
    max_x = 1267519

    # round to the nearest 1000th
    min_y = np.floor(min_y / 1000) * 1000.0
    max_y = np.ceil(max_y / 1000) * 1000.0
    min_x = np.floor(min_x / 1000) * 1000.0
    max_x = np.ceil(max_x / 1000) * 1000.0

    # make a regular grid of points
    x = np.arange(min_x, max_x, resolution)
    y = np.arange(min_y, max_y, resolution)

    XC = np.zeros((len(y), len(x)))
    YC = np.zeros((len(y), len(x)))
    for i in range(len(y)):
        XC[i, :] = x
    for j in range(len(x)):
        YC[:, j] = y

    # plt.plot(x)
    # plt.title('x')
    # plt.show()
    #
    # plt.plot(y)
    # plt.title('y')
    # plt.show()

    # XC, YC = np.meshgrid(x, y)

    # fig = plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.pcolormesh(XC, cmap='turbo')
    # plt.colorbar()
    # plt.title('XC')
    # plt.subplot(1, 2, 2)
    # plt.pcolormesh(YC, cmap='turbo')
    # plt.colorbar()
    # plt.title('YC')
    # plt.show()

    # get the remainder when dividing the grid size by sNx and sNy
    x_remainder = len(x) % sNx
    y_remainder = len(y) % sNy
    print('    - x_remainder: ', x_remainder)
    print('    - y_remainder: ', y_remainder)
    print('    - x size: ', len(x))
    print('    - y size: ', len(y))

    x = np.arange(min_x, max_x + (sNx - x_remainder) * resolution, resolution)
    y = np.arange(min_y, max_y + (sNx - y_remainder) * resolution, resolution)
    XC, YC = np.meshgrid(x, y)

    XC = np.zeros((len(y), len(x)))
    YC = np.zeros((len(y), len(x)))
    for i in range(len(y)):
        XC[i, :] = x
    for j in range(len(x)):
        YC[:, j] = y

    # plt.plot(x)
    # plt.title('x')
    # plt.show()
    #
    # plt.plot(y)
    # plt.title('y')
    # plt.show()

    xg = np.arange(min_x-resolution/2, max_x + (sNx - x_remainder) * resolution + resolution/2, resolution)
    yg = np.arange(min_y-resolution/2, max_y + (sNx - y_remainder) * resolution + resolution/2, resolution)
    XG, YG = np.meshgrid(xg, yg)

    # XG = np.copy(XC)-resolution/2
    # YG = np.copy(YC)-resolution/2

    print('    - XC shape: ', np.shape(XC))
    print('    - YC shape: ', np.shape(YC))
    print('    - XG shape: ', np.shape(XG))
    print('    - YG shape: ', np.shape(YG))

    if len(x)%sNx != 0:
        raise ValueError('Grid size of '+str(len(x))+' is not divisible by sNx')
    if len(y)%sNy != 0:
        raise ValueError('Grid size of '+str(len(y))+' is not divisible by sNy')
    return(XC, YC, XG, YG)

def plot_grid(project_folder, XC, YC, XG, YG, Bathy, XC_4326, YC_4326, XG_4326, YG_4326,
              sNx, sNy, resolution):

    fig = plt.figure(figsize=(14,8))

    plt.subplot(1, 2, 1)
    plt.pcolormesh(XC, YC, Bathy, cmap='Blues', shading='auto', vmin=0, vmax=2000, alpha=0.5)

    nPx = np.shape(XC)[1]/sNx
    nPy = np.shape(XC)[0]/sNy

    for n in range(1,int(nPx)):
        plt.plot([XC[0, n*sNx], XC[-1, n*sNx]], [YC[0, n*sNx], YC[-1, n*sNx]], color='k', lw=0.5)
    for n in range(1,int(nPy)):
        plt.plot([XC[n*sNy, 0], XC[n*sNy, -1]], [YC[n*sNy, 0], YC[n*sNy, -1]], color='k', lw=0.5)

    blank_cells = 0
    non_blank_cells = 0
    for px in range(int(nPx)):
        for py in range(int(nPy)):
            bathy_subset = Bathy[py*sNy:(py+1)*sNy, px*sNx:(px+1)*sNx]
            if np.any(bathy_subset > 0):
                non_blank_cells += 1
                # plot a double hatched polygon
                # poly = Polygon([[X[py * sNy, px * sNx], Y[py * sNy, px * sNx]],
                #                 [X[py * sNy, (px + 1) * sNx-1], Y[py * sNy, (px + 1) * sNx-1]],
                #                 [X[(py + 1) * sNy-1, (px + 1) * sNx-1], Y[(py + 1) * sNy-1, (px + 1) * sNx-1]],
                #                 [X[(py + 1) * sNy-1, px * sNx], Y[(py + 1) * sNy-1, px * sNx]]],
                #                closed=True, fill=False, hatch='/', edgecolor='k', lw=0.5)
                # plt.gca().add_patch(poly)
            else:
                blank_cells += 1
                poly = Polygon([[XC[py * sNy, px * sNx], YC[py * sNy, px * sNx]],
                                [XC[py * sNy, (px + 1) * sNx-1]+resolution, YC[py * sNy, (px + 1) * sNx-1]],
                                [XC[(py + 1) * sNy-1, (px + 1) * sNx-1]+resolution, YC[(py + 1) * sNy-1, (px + 1) * sNx-1]+resolution],
                                [XC[(py + 1) * sNy-1, px * sNx], YC[(py + 1) * sNy-1, px * sNx]+resolution]],
                               closed=True, fill=False, hatch='//', edgecolor='k', lw=0.5)
                plt.gca().add_patch(poly)
            # plt.plot(X[py*sNy, px*sNx], Y[py*sNy, px*sNx], 'ko', markersize=3)

    plt.title('Grid Size: '+str(np.shape(XC)[0])+' x '+str(np.shape(XC)[1])+',    Resolution: '+str(resolution)+' m'+\
              '\n sNx = '+str(sNx)+'  sNy = '+str(sNy)+',   Total cells = '+str(blank_cells+non_blank_cells)+\
                '\n Blank cells: '+str(blank_cells)+'  Non-blank cells: '+str(non_blank_cells)+\
              '\n Total Broadwell nodes: '+str(int(np.ceil((blank_cells+non_blank_cells)/28))),
               fontsize=12)

    plt.xlim([np.min(XC), np.max(XC)])
    plt.ylim([np.min(YC), np.max(YC)])

    plt.subplot(2, 4, 3)
    C = plt.pcolormesh(XC, YC, XC_4326, shading='auto')
    plt.colorbar(C, orientation='horizontal', pad=0.1)
    plt.title('XC')

    plt.subplot(2, 4, 4)
    C = plt.pcolormesh(XC, YC, YC_4326, shading='auto')
    plt.colorbar(C, orientation='horizontal', pad=0.1)
    plt.title('YC')

    plt.subplot(2, 4, 7)
    C = plt.pcolormesh(XG, YG, XG_4326, shading='auto')
    plt.colorbar(C, orientation='horizontal', pad=0.1)
    plt.title('XG')

    plt.subplot(2, 4, 8)
    C = plt.pcolormesh(XG, YG, YG_4326, shading='auto')
    plt.colorbar(C, orientation='horizontal', pad=0.1)
    plt.title('YG')

    plt.savefig(os.path.join(project_folder,'Figures', 'model_grid_'+str(resolution)+'.png'), dpi=300)
    plt.close(fig)

project_folder = '/Users/mhwood/Documents/Research/Projects/Chukchi_Sea'

sNx = 60
sNy = 60
resolution = 2000.0

print(' - Making the model grid')
# make the grid
XC, YC, XG, YG = create_grid(sNx, sNy, resolution)

print('XC',np.min(XC), np.max(XC))
print('YC',np.min(YC), np.max(YC))

# fig = plt.figure(figsize=(12, 8))
# plt.subplot(2, 2, 1)
# plt.pcolormesh(XC, shading='nearest', cmap='turbo')
# plt.colorbar()
# plt.title('XC')
# plt.subplot(2, 2, 2)
# plt.pcolormesh(YC, shading='nearest', cmap='turbo')
# plt.colorbar()
# plt.title('YC')
# plt.subplot(2, 2, 3)
# plt.pcolormesh(XG, shading='nearest', cmap='turbo')
# plt.colorbar()
# plt.title('XG')
# plt.subplot(2, 2, 4)
# plt.pcolormesh(YG, shading='nearest', cmap='turbo')
# plt.colorbar()
# plt.title('YG')
# plt.suptitle('Before reprojection')
# plt.show()


# reproject the 4 grids above to 4326
print(' - Reprojecting the grid')
# reproject XC and YC
grid_points = np.column_stack([XC.ravel(), YC.ravel()])
grid_points = reproject_polygon(grid_points, 32602, 4326)
XC_4326 = grid_points[:,0].reshape(XC.shape)
XC_4326[XC_4326<0] += 360  # ensure all longitudes are positive
YC_4326 = grid_points[:,1].reshape(YC.shape)
# reproject XG and YG
grid_points = np.column_stack([XG.ravel(), YG.ravel()])
grid_points = reproject_polygon(grid_points, 32602, 4326)
XG_4326 = grid_points[:,0].reshape(XG.shape)
XG_4326[XG_4326<0] += 360  # ensure all longitudes are positive
YG_4326 = grid_points[:,1].reshape(YG.shape)

print('XC',np.min(XC_4326), np.max(XC_4326))
print('YC',np.min(YC_4326), np.max(YC_4326))

print(' - Reading in the bathymetry')
# read in the bathymetry
bathy_lon, bathy_lat, bathy = read_coarse_bathymetry_from_nc()

print(np.min(bathy_lon), np.max(bathy_lon))
print(np.min(bathy_lat), np.max(bathy_lat))

# plt.pcolormesh(bathy, cmap='Blues', shading='auto')
# plt.show()

# 60.00,170.24
# 72.25,-147.11

test_points = reproject_polygon(np.array([[-534000.0, 6800000.0],
                                          [1384000.0, 8238000.0]]), 32602, 4326)
print('test points reprojection: ', test_points)

test_points = reproject_polygon(np.array([[170.28951038, 60.00356107],
                                          [-143.96487232, 72.41065205]]), 4326, 32602)
print('test points reprojection: ', test_points)

print(' - Reprojecting the bathymetry grid')
# rerpoject the bathymetry data
bathy_Lon, bathy_Lat = np.meshgrid(bathy_lon, bathy_lat)
# bathy_Lon[bathy_Lon<0] += 360  # ensure all longitudes are positive
bathy_points = np.column_stack([bathy_Lon.ravel(), bathy_Lat.ravel()])
bathy_points = reproject_polygon(bathy_points, 4326, 32602)
bathy_X = bathy_points[:,0].reshape(bathy_Lon.shape)
bathy_Y = bathy_points[:,1].reshape(bathy_Lon.shape)

print(np.min(bathy_X), np.max(bathy_X))
print(np.min(bathy_Y), np.max(bathy_Y))

# fig = plt.figure(figsize=(12, 8))
# plt.subplot(2, 2, 1)
# plt.pcolormesh(XC_4326, shading='nearest', cmap='turbo')
# plt.colorbar()
# plt.title('XC')
# plt.subplot(2, 2, 2)
# plt.pcolormesh(YC_4326, shading='nearest', cmap='turbo')
# plt.colorbar()
# plt.title('YC')
# plt.subplot(2, 2, 3)
# plt.pcolormesh(XG_4326, shading='nearest', cmap='turbo')
# plt.colorbar()
# plt.title('XG')
# plt.subplot(2, 2, 4)
# plt.pcolormesh(YG_4326, shading='nearest', cmap='turbo')
# plt.colorbar()
# plt.title('YG')
# plt.suptitle('After reprojection')
# plt.show()
#
# print(np.shape(bathy_X), np.shape(bathy_Y), np.shape(bathy))
# plt.pcolormesh(bathy_Lon, bathy_Lat, bathy, cmap='Blues', shading='auto', vmin=0, vmax=50)
# plt.xlim([np.min(XC_4326), np.max(XC_4326)])
# plt.ylim([np.min(YC_4326), np.max(YC_4326)])
# plt.show()

print(' - Interpolating the bathymetry onto the grid')
# interpolate the bathymetry onto the grid
Bathy = griddata((bathy_X.flatten(), bathy_Y.flatten()), bathy.flatten(), (XC, YC), method='linear')

# plt.pcolormesh(Bathy, cmap='Blues', shading='auto', vmin=0, vmax=2000)
# plt.colorbar()
# plt.show()

plot_grid(project_folder, XC, YC, XG, YG, Bathy, XC_4326, YC_4326, XG_4326, YG_4326,
          sNx, sNy, resolution)

print('    - XC_4326 shape: ', np.shape(XC_4326))
print('    - YC_4326 shape: ', np.shape(YC_4326))
print('    - XG_4326 shape: ', np.shape(XG_4326))
print('    - YG_4326 shape: ', np.shape(YG_4326))

# save the grid
print(' - Saving the grid')
lon_file = os.path.join(project_folder, 'Data', 'XC.data')
np.array(XC_4326).ravel('C').astype('>f4').tofile(lon_file)
lat_file = os.path.join(project_folder, 'Data', 'YC.data')
np.array(YC_4326).ravel('C').astype('>f4').tofile(lat_file)
lon_file = os.path.join(project_folder, 'Data', 'XG.data')
np.array(XG_4326).ravel('C').astype('>f4').tofile(lon_file)
lat_file = os.path.join(project_folder, 'Data', 'YG.data')
np.array(YG_4326).ravel('C').astype('>f4').tofile(lat_file)

# XC = np.fromfile(os.path.join(project_folder, 'Data', 'XC.data'), dtype='>f4').reshape((1590, 1050))
# YC = np.fromfile(os.path.join(project_folder, 'Data', 'YC.data'), dtype='>f4').reshape((1590, 1050))
# XG = np.fromfile(os.path.join(project_folder, 'Data', 'XG.data'), dtype='>f4').reshape((1591, 1051))
# YG = np.fromfile(os.path.join(project_folder, 'Data', 'YG.data'), dtype='>f4').reshape((1591, 1051))
#
# fig = plt.figure(figsize=(12, 8))
# plt.subplot(2, 2, 1)
# plt.pcolormesh(XC, shading='nearest', cmap='turbo')
# plt.colorbar()
# plt.title('XC')
# plt.subplot(2, 2, 2)
# plt.pcolormesh(YC, shading='nearest', cmap='turbo')
# plt.colorbar()
# plt.title('YC')
# plt.subplot(2, 2, 3)
# plt.pcolormesh(XG, shading='nearest', cmap='turbo')
# plt.colorbar()
# plt.title('XG')
# plt.subplot(2, 2, 4)
# plt.pcolormesh(YG, shading='nearest', cmap='turbo')
# plt.colorbar()
# plt.title('YG')
# plt.show()
