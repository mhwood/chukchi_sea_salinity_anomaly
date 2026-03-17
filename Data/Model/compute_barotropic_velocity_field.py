
import os
import numpy as np
import netCDF4 as nc4
import matplotlib.pyplot as plt

def read_model_grid(config_dir):
    grid_file = os.path.join(config_dir, 'Chukchi_Sea_grid.nc')
    ds = nc4.Dataset(grid_file)

    lon = ds.variables['XC'][:,:]
    lat = ds.variables['YC'][:,:]
    depth = ds.variables['Depth'][:,:]
    hFacC = ds.variables['HFacC'][:,:,:]
    drF = ds.variables['drF'][:]

    ds.close()

    return lon, lat, depth, hFacC, drF

def read_mean_velocity_field_from_nc_files(config_dir):

    mean_var_fields = {}

    for var_name in ['Uvel', 'Vvel']:

        mean_velocity_grid = np.zeros((90,720,960))
        count = 0

        for month in range(7,13):
            file_path = os.path.join(config_dir, 'results_control', 'daily_mean', var_name,
                                     f'{var_name}_2023{month:02d}.nc')
            print(f'  - Reading {var_name}_2023{month:02d}.nc')

            ds = nc4.Dataset(file_path)
            vel_grid = ds.variables[var_name][:,:,:]
            ds.close()

            for timestep in range(np.shape(vel_grid)[0]):
                mean_velocity_grid += vel_grid[timestep,:,:]
                count += 1

        mean_velocity_grid /= count

        mean_var_fields[var_name] = mean_velocity_grid

    return mean_var_fields

def compute_barotropic_velocity(mean_var_fields, hFacC, drF):
    # barotropic velocity is the vertically integrated velocity
    Uvel = mean_var_fields['Uvel']
    Vvel = mean_var_fields['Vvel']
    barotropic_U = np.zeros((np.shape(Uvel)[1], np.shape(Uvel)[2]))
    barotropic_V = np.zeros((np.shape(Vvel)[1], np.shape(Vvel)[2]))
    for row in range(np.shape(Uvel)[1]):
        for col in range(np.shape(Uvel)[2]):
            if hFacC[0,row,col]>0:
                barotropic_U[row, col] = np.sum(Uvel[:, row, col] * hFacC[:, row, col] * drF) / np.sum(hFacC[:, row, col] * drF)
                barotropic_V[row, col] = np.sum(Vvel[:, row, col] * hFacC[:, row, col] * drF) / np.sum(hFacC[:, row, col] * drF)
    return barotropic_U, barotropic_V

def write_barotropic_velocity_to_nc(project_dir, XC, YC, barotropic_U, barotropic_V):
    output_file = os.path.join(project_dir,'Data', 'Model', 'barotropic_velocity_spinup.nc')
    with nc4.Dataset(output_file, 'w', format='NETCDF4') as ds:
        ds.createDimension('x', barotropic_U.shape[1])
        ds.createDimension('y', barotropic_U.shape[0])

        x_var = ds.createVariable('Longitude', 'f8', ('y', 'x'))
        y_var = ds.createVariable('Latitude', 'f8', ('y', 'x'))
        u_var = ds.createVariable('Uvel', 'f8', ('y', 'x'))
        v_var = ds.createVariable('Vvel', 'f8', ('y', 'x'))

        x_var[:] = XC
        y_var[:] = YC
        u_var[:] = barotropic_U
        v_var[:] = barotropic_V

        x_var.units = 'degrees_east'
        y_var.units = 'degrees_north'
        u_var.units = 'm/s'
        v_var.units = 'm/s'


config_dir = '/Volumes/kullorsuaq/Research/Ocean_Modeling/Projects/Chukchi_Sea'

project_dir = '/Users/mike/Documents/Research/Projects/Chukchi Sea'

mean_var_fields = {}

lon, lat, depth, hFacC, drF = read_model_grid(config_dir)

print('Reading mean velocity fields from netCDF files...')
mean_var_fields = read_mean_velocity_field_from_nc_files(config_dir)

print('Computing barotropic velocity...')
barotropic_U, barotropic_V = compute_barotropic_velocity(mean_var_fields, hFacC, drF)

print('Writing barotropic velocity to netCDF file...')
write_barotropic_velocity_to_nc(project_dir, lon, lat, barotropic_U, barotropic_V)

