
import os
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc4
from scipy.interpolate import griddata
from pyproj import Proj, Transformer

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
    elif inputCRS == 3411 and outputCRS == 32602:
        x2, y2 = transformer.transform(polygon_array[:, x_column], polygon_array[:, y_column])
        x2 = np.array(x2)
        y2 = np.array(y2)
    else:
        raise ValueError('Reprojection with this epsg is not safe - no test for validity has been implemented')

    output_polygon=np.copy(polygon_array)
    output_polygon[:,x_column] = x2
    output_polygon[:,y_column] = y2
    return output_polygon

def read_grid_geometry_from_nc(config_dir, model_name):
    file_path = os.path.join(config_dir, 'nc_grids', model_name + '_grid.nc')
    ds = nc4.Dataset(file_path)
    XC = ds.variables['XC'][:,:]
    YC = ds.variables['YC'][:,:]
    Depth = ds.variables['Depth'][:,:]
    ds.close()

    points = np.column_stack([XC.ravel(), YC.ravel()])
    points = reproject_polygon(points, 4326, 32602)
    X_32602 = np.reshape(points[:, 0], np.shape(XC))
    Y_32602 = np.reshape(points[:, 1], np.shape(YC))

    return(XC, YC, X_32602, Y_32602, Depth)

def read_annual_data_stack_to_model_grid(data_dir, year, XC, YC, X_32602, Y_32602, Depth):

    first_file = True
    if year%4==0:
        seaice_stack = np.zeros((366,np.shape(X_32602)[0], np.shape(X_32602)[1]))
    else:
        seaice_stack = np.zeros((365, np.shape(X_32602)[0], np.shape(X_32602)[1]))

    counter = 0

    for month in range(1, 13):
        print('  - Reading in data in month '+str(month)+' in year '+str(year))
        if month in [1, 3, 5, 7, 8, 10, 12]:
            n_days = 31
        elif month in [4, 6, 9, 11]:
            n_days = 30
        else:
            if year % 4 == 0:
                n_days = 29
            else:
                n_days = 28

        yr = str(year)
        mo = '{:02d}'.format(month)

        for day in range(1,n_days+1):
            # try:
            dy = '{:02d}'.format(day)
            if year>=2008:
                file_path = os.path.join(data_dir, 'seaice_conc_daily_nh_'+yr+mo+dy+'_f17_v04r00.nc')
            else:
                file_path = os.path.join(data_dir, 'seaice_conc_daily_nh_' + yr + mo + dy + '_f13_v04r00.nc')
            print('       - Reading '+'seaice_conc_daily_nh_'+yr+mo+dy+'_f17_v04r00.nc')

            ds = nc4.Dataset(file_path)
            if first_file:
                x = ds.variables['xgrid'][:]
                y = ds.variables['ygrid'][:]
            seaice = ds.variables['nsidc_nt_seaice_conc'][:,:,:]
            ds.close()
            seaice = np.array(seaice[0, :, :])

            if first_file:
                X_3411,Y_3411 = np.meshgrid(x,y)
                # plt.pcolormesh(X_3411, Y_3411, seaice)
                # plt.show()

                points = np.column_stack([X_3411.ravel(), Y_3411.ravel()])
                points = reproject_polygon(points, 3411, 32602)
                X_32602_seaice = np.reshape(points[:, 0],np.shape(X_3411))
                Y_32602_seaice = np.reshape(points[:, 1], np.shape(Y_3411))

                first_file = False

            # fill the inland points to avoid interpolation issues?
            points_nonnan = np.column_stack([X_3411.ravel(),Y_3411.ravel()])
            seaice_nonnan = np.ravel(seaice)
            non_zero_locations = seaice_nonnan<2
            points_nonnan = points_nonnan[non_zero_locations,:]
            seaice_nonnan = seaice_nonnan[non_zero_locations]
            seaice_nearest = griddata(points_nonnan, seaice_nonnan, (X_3411, Y_3411),
                                      method='nearest')
            # plt.subplot(1, 2, 1)
            # C = plt.pcolormesh(seaice, vmin=0, vmax=1)
            # plt.colorbar(C, orientation='horizontal')
            # plt.subplot(1, 2, 2)
            # C=plt.pcolormesh(seaice_nearest)
            # plt.colorbar(C, orientation='horizontal')
            # plt.show()

            seaice_32602= griddata(np.column_stack([X_32602_seaice.ravel(),Y_32602_seaice.ravel()]),
                                  seaice_nearest.ravel(), (X_32602, Y_32602))
            seaice_32602[Depth<=0]=254

            seaice_stack[counter,:,:] = seaice_32602

            # plt.subplot(1, 2, 1)
            # C = plt.pcolormesh(X_32602_seaice, Y_32602_seaice, seaice,vmin=0,vmax=1)
            # plt.gca().set_xlim([np.min(X_32602), np.max(X_32602)])
            # plt.gca().set_ylim([np.min(Y_32602), np.max(Y_32602)])
            # plt.colorbar(C, orientation='horizontal')
            # plt.subplot(1, 2, 2)
            # C=plt.pcolormesh(X_32602, Y_32602, seaice_32602,vmin=0,vmax=1)
            # plt.colorbar(C, orientation='horizontal')
            # plt.show()
            # except:
            #     print('   There was an issue with this file')

            counter +=1

    return(seaice_stack)

def write_data_to_annual_nc(output_file,X,Y,seaice):

    ds= nc4.Dataset(output_file,'w')
    tdim = ds.createDimension('time',np.shape(seaice)[0])
    ydim = ds.createDimension('y',np.shape(seaice)[1])
    xdim = ds.createDimension('x', np.shape(seaice)[2])

    x = ds.createVariable('x','f4',('x',))
    x[:] = X[0,:]

    y = ds.createVariable('y', 'f4', ('y',))
    y[:] = Y[:,0]

    d = ds.createVariable('days', 'f4', ('time',))
    d[:] = np.arange(1,np.shape(seaice)[0]+1)

    s = ds.createVariable('seaice_conc', 'f4', ('time','y', 'x'))
    s[:,:,:] = seaice

    ds.close()

def read_data_from_annual_nc(input_file):

    ds = nc4.Dataset(input_file, 'r')
    seaice = ds.variables['seaice_conc'][:,:,:]
    x = ds.variables['x'][:]
    y = ds.variables['y'][:]
    days = ds.variables['days'][:]
    ds.close()

    return seaice, x, y, days


config_dir = '/Users/mhwood/Documents/Research/Projects/Ocean_Modelling/Projects/Downscale_Greenland/' \
             'MITgcm/configurations/downscale_greenland'

project_dir = '/Users/mhwood/Documents/Research/Projects/Chukchi_Sea'
# project_dir = '/Users/mike/Documents/Research/Projects/Greenland Model Analysis/Fjord/Disko Bay'

data_dir = '/Volumes/CoOL/Data_Repository/Arctic/Sea_Ice/daily'

XC, YC, X_32602, Y_32602, Depth = read_grid_geometry_from_nc(config_dir, model_name='Chukchi_Sea')

for year in range(2023, 2025):

    file_name = 'Chukchi Sea SeaIice '+str(year)+'.nc'

    output_file = os.path.join(project_dir, 'Data', 'Sea Ice', file_name)

    if file_name not in os.listdir(os.path.join(project_dir,'Data','Sea Ice')):
        print('Stacking data in year ' + str(year))

        seaice_stack = read_annual_data_stack_to_model_grid(data_dir, year, XC, YC, X_32602, Y_32602, Depth)

        write_data_to_annual_nc(output_file,X_32602, Y_32602,seaice_stack)

    # read the data back in
    seaice, x, y, days = read_data_from_annual_nc(output_file)
    seaice = np.array(seaice)

    # write the data to model binary
    bin_file = os.path.join(project_dir, 'Model', 'input', 'exf','Chukchi_Seaice_' + str(year))
    seaice.ravel('C').astype('>f4').tofile(bin_file)