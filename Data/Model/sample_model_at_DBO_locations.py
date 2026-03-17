
import os
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc4
from pyproj import Transformer
from scipy.interpolate import griddata
import cmocean.cm as cm
from matplotlib.patches import Rectangle
from datetime import timedelta, datetime
# import Gridspec
from matplotlib.gridspec import GridSpec
# ignore UserWarning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import datetime

def YMD_to_DecYr(year,month,day,hour=0,minute=0,second=0):
    date = datetime.datetime(year,month,day,hour,minute,second)
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
    decimal_fraction = float(date.toordinal() - start) / year_length
    dec_yr = year+decimal_fraction
    return(dec_yr)

def get_DBO_coordinates(dbo_section):

    if dbo_section == 'BS':
        points = np.array([[ 191.849 , 65.61883333333333 ],
                            [ 191.729 , 65.64783333333334 ],
                            [ 191.63433333333333 , 65.67033333333333 ],
                            [ 191.53966666666668 , 65.69266666666667 ],
                            [ 191.445 , 65.71516666666666 ],
                            [ 191.35016666666667 , 65.7375 ],
                            [ 191.25516666666667 , 65.76 ],
                            [ 191.16016666666667 , 65.78233333333333 ],
                            [ 190.929 , 65.8415 ],
                            [ 190.858 , 65.86266666666667 ],
                            [ 190.786 , 65.88 ],
                            [ 190.721 , 65.89283333333333 ],
                            [ 190.649 , 65.91 ],
                            [ 190.576 , 65.92816666666667 ],
                            [ 190.504 , 65.94533333333334 ],
                            [ 190.434 , 65.96233333333333 ],
                            [ 190.35783333333333 , 65.98166666666667 ],])


    if dbo_section == 'DBO3':
        points = np.array([[ 193.166 , 68.323 ],
                    [ 193.079 , 68.30133333333333 ],
                    [ 192.891 , 68.2435 ],
                    [ 192.701 , 68.19116666666666 ],
                    [ 192.516 , 68.129 ],
                    [ 192.325 , 68.07266666666666 ],
                    [ 192.139 , 68.0165 ],
                    [ 191.955 , 67.96066666666667 ],
                    [ 191.767 , 67.904 ],
                    [ 191.586 , 67.8435 ],
                    [ 191.405 , 67.78566666666667 ],
                    [ 191.222 , 67.72866666666667 ],
                    [ 191.05 , 67.675 ],
                    [ 190.68183333333334 , 67.56683333333334 ],
                    [ 190.313 , 67.45516666666667 ],
                    [ 189.94616666666667 , 67.3435 ],
                    [ 189.58083333333335 , 67.23183333333333 ],
                    [ 189.21733333333333 , 67.12016666666666 ],
                    [ 188.85533333333333 , 67.00866666666667 ],
                    [ 188.49516666666668 , 66.897 ]])

    if dbo_section == 'DBO5':
        points = np.array([[ 202.933 , 71.19166666666666 ],
                    [ 202.88983333333334 , 71.21583333333334 ],
                    [ 202.835 , 71.247 ],
                    [ 202.752 , 71.288 ],
                    [ 202.668 , 71.33 ],
                    [ 202.585 , 71.372 ],
                    [ 202.51 , 71.41 ],
                    [ 202.417 , 71.455 ],
                    [ 202.34 , 71.5 ],
                    [ 202.247 , 71.537 ],
                    [ 202.162 , 71.578 ],
                    [ 202.075 , 71.62 ]])

    return points

def read_model_grid(project_folder):
    ds = nc4.Dataset(os.path.join(project_folder,'Data', 'Model', 'Chukchi_Sea_grid.nc'))
    XC = ds.variables['XC'][:,:]
    YC = ds.variables['YC'][:,:]
    Depth = ds.variables['Depth'][:,:]
    drF = ds.variables['drF'][:]
    ds.close()
    Z_bottom = np.cumsum(drF)
    Z_top = Z_bottom - drF
    Z = 0.5 * (Z_bottom + Z_top)
    return(XC, YC, Depth, Z)

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

def var_name_to_subset_and_filename(var_name, date_str):
    if var_name in ['Theta','Salt','Uvel','Vvel']:
        subset = 'daily_mean'
        file_name = var_name + '_' + date_str[:6] + '.nc'
    elif var_name in ['SIheff','SIarea']:
        subset = 'daily_snapshot'
        file_name = var_name + '_' + date_str[:6] + '.nc'
    return subset, file_name

def read_model_field_from_nc(config_dir, results_dir, subset, var_name, year, month):

    if var_name in ['Theta','Salt','Uvel','Vvel']:
        subset = 'daily_mean'
        file_name = var_name + '_' + f'{year}{month:02d}' + '.nc'
    elif var_name in ['SIheff','SIarea']:
        subset = 'daily_snapshot'
        file_name = var_name + '_' + f'{year}{month:02d}' + '.nc'

    ds = nc4.Dataset(os.path.join(config_dir, results_dir, subset, var_name, file_name))
    if var_name in ['Theta', 'Salt','Uvel','Vvel']:
        data = ds.variables[var_name][:,:,:,:]
    elif var_name in ['SIheff','SIarea']:
        data = ds.variables[var_name][:,:,:]
    ds.close()

    return data

def interpolate_field_onto_DBO_section(field_data, X, Y, Z, days_in_month, dbo_point_set):

    interpolated_subsets = {}

    for section_name in dbo_point_set.keys():
        dbo_points = dbo_point_set[section_name]

        # get subset of field data just around the DBO section
        min_x = np.min(dbo_points[:,0]) - 5000
        max_x = np.max(dbo_points[:,0]) + 5000
        min_y = np.min(dbo_points[:,1]) - 5000
        max_y = np.max(dbo_points[:,1]) + 5000
        x_mask = np.logical_and(X[0,:] >= min_x, X[0,:] <= max_x)
        y_mask = np.logical_and(Y[:,0] >= min_y, Y[:,0] <= max_y)
        X_subset = X[y_mask,:][:,x_mask]
        Y_subset = Y[y_mask,:][:,x_mask]
        field_data_subset = field_data[:, :, y_mask, :][:, :, :, x_mask]
        field_data_subset[field_data_subset==0] = np.nan

        interpolated_grid = np.zeros((days_in_month, len(Z), dbo_points.shape[0]))
        print(f'                - Interpolating section {section_name}')

        for day in range(days_in_month):


            for z in range(len(Z)):
                grid_z = griddata((X_subset.flatten(), Y_subset.flatten()), field_data_subset[day, z, :, :].flatten(),
                                  (dbo_points[:,0], dbo_points[:,1]), method='linear')
                interpolated_grid[day,z,:] = grid_z

        interpolated_subsets[section_name] = interpolated_grid

    return interpolated_subsets

def write_subsets_to_nc(project_folder, var_name, experiment, time, interpolated_subsets, Z):

    subset_file = var_name + '_DBO_Sections_' + experiment + '.nc'
    dataset = nc4.Dataset(os.path.join(project_folder, 'Data', 'Model', 'DBO Sections', subset_file), 'w')

    days_dim = dataset.createDimension('dec_yrs', np.shape(interpolated_subsets[list(interpolated_subsets.keys())[0]])[0])
    depth_dim = dataset.createDimension('depth', len(Z))

    for section_name in interpolated_subsets.keys():

        grp = dataset.createGroup(section_name)

        point_dim = grp.createDimension('points', interpolated_subsets[section_name].shape[2])

        days_var = grp.createVariable('dec_yrs', np.float32, ('dec_yrs',))
        depth_var = grp.createVariable('depth', np.float32, ('depth',))
        data_var = grp.createVariable(var_name, np.float32, ('dec_yrs', 'depth', 'points'))

        depth_var.units = 'meters'

        days_var[:] = time
        depth_var[:] = Z
        data_var[:, :, :] = interpolated_subsets[section_name]

    dataset.close()

project_folder = '/Users/mike/Documents/Research/Projects/Chukchi Sea/'

XC, YC, Depth, Z = read_model_grid(project_folder)

# reproject the model grid to 32602
points = reproject_points(np.column_stack((XC.flatten(), YC.flatten())), 4326, 32602)
X = points[:, 0].reshape(XC.shape)
Y = points[:, 1].reshape(YC.shape)

config_dir = '/Volumes/chukchi_sea/Research/Ocean_Modeling/Projects/Chukchi_Sea'

dbo_point_set = {}
dbo_point_set['BS'] = reproject_points(get_DBO_coordinates('BS'), 4326, 32602)
dbo_point_set['DBO3'] = reproject_points(get_DBO_coordinates('DBO3'), 4326, 32602)
dbo_point_set['DBO5'] = reproject_points(get_DBO_coordinates('DBO5'), 4326, 32602)

for results_dir in ['results_daily_hydrology', 'results_control','results_prescribed_seaice']:
    print('Processing results for the ' + results_dir+' experiment')
    for var_name in ['Vvel','Theta','Salt','SIheff','SIarea']:
        print('    - Creating subset for ' + var_name)

        first_file = True
        experiment = results_dir.replace('results_', '')
        subset_file = var_name + '_DBO_Sections_' + experiment + '.nc'
        if subset_file not in []:#os.listdir(os.path.join(project_folder, 'Data', 'Model', 'DBO Sections')):

            for year in [2023, 2024]:

                print('        - Processing year ' + str(year))
                all_file_paths = []
                for month in range(1,13):
                    print('            - Processing month ' + f'{month:02d}')
                    if month in [1,3,5,7,8,10,12]:
                        days_in_month = 31
                    elif month in [4,6,9,11]:
                        days_in_month = 30
                    elif month == 2:
                        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                            days_in_month = 29
                        else:
                            days_in_month = 28
                    if month ==12 and year ==2024:
                        days_in_month = 29
                    month_decyrs = np.array([YMD_to_DecYr(year, month, day+1) for day in range(days_in_month)])

                    subset, file_name = var_name_to_subset_and_filename(var_name, f'{year}{month:02d}')

                    if file_name in os.listdir(os.path.join(config_dir, results_dir, subset, var_name)):
                        print('                - Reading file ' + file_name)
                        field_data = read_model_field_from_nc(config_dir, results_dir, subset, var_name, year, month)
                        interpolated_subsets = interpolate_field_onto_DBO_section(field_data, X, Y, Z, days_in_month, dbo_point_set)

                        if first_file:
                            combined_interpolated_subsets = interpolated_subsets
                            combined_month_decyrs = month_decyrs
                            first_file = False
                        else:
                            for section_name in interpolated_subsets.keys():
                                combined_interpolated_subsets[section_name] = np.concatenate((combined_interpolated_subsets[section_name],
                                                                                             interpolated_subsets[section_name]), axis=0)
                            combined_month_decyrs = np.concatenate((combined_month_decyrs, month_decyrs), axis=0)

            write_subsets_to_nc(project_folder, var_name, experiment, combined_month_decyrs, combined_interpolated_subsets, Z)




