
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
    ds.close()
    return XC, YC, Depth

def read_precip_data(era_xx_folder,year):
    lon = np.arange(0, 360, 0.7031250)

    del_lat = np.concatenate([np.array([0]),
                              np.array([0.6958694]),
                              np.array([0.6999817]),
                              np.array([0.7009048]),
                              np.array([0.7012634]),
                              np.array([0.7014313]),
                              0.7017418 * np.ones((245,)),
                              np.array([0.7014313]),
                              np.array([0.7012634]),
                              np.array([0.7009048]),
                              np.array([0.6999817]),
                              np.array([0.6958694]),
                              ])
    lat = np.cumsum(del_lat) + -89.4628220

    grid = np.fromfile(os.path.join(era_xx_folder, 'EIG_rain_plus_ECCO_v4r1_ctrl_' + str(year)), '>f4')
    n_timesteps = int(np.size(grid) / (len(lon) * len(lat)))
    grid = np.reshape(grid, (n_timesteps, len(lat), len(lon)))

    # convert m/s to mm/day
    grid *= 86400 * 1000  # Convert from m/s to mm/day

    return(lon, lat, grid)

def read_precip_to_model_grid(X, Y, precip_X, precip_Y, precip_data, testing):

    precip_grid = np.zeros((np.shape(precip_data)[0], np.shape(X)[0], np.shape(X)[1]))

    skip = 100

    if testing:
        for i in range(0,np.shape(precip_data)[0],skip):
            if i%100==0:
                print('    - Interpolating timestep {}/{}'.format(i, np.shape(precip_data)[0]))
            interp = griddata((precip_X.flatten(), precip_Y.flatten()), precip_data[i,:,:].flatten(),
                                           (X, Y), method='linear')
            for j in range(skip):
                if i+j<np.shape(precip_data)[0]:
                    precip_grid[i+j,:,:] = interp
    else:
        for i in range(np.shape(precip_data)[0]):
            if i%100==0:
                print('    - Interpolating timestep {}/{}'.format(i, np.shape(precip_data)[0]))
            precip_grid[i,:,:] = griddata((precip_X.flatten(), precip_Y.flatten()), precip_data[i,:,:].flatten(),
                                           (X, Y), method='linear')

    return precip_grid

def read_Blaskey_runoff_data(project_folder):
    ds = nc4.Dataset(os.path.join(project_folder, 'Data', 'Observations','River Discharge',
                                  'Alaska_River_discharge_in_Chukchi_Sea_domain.nc'))
    runoff = ds.variables['runoff'][:,:]
    ds.close()

    plt.plot(runoff[15,:])
    plt.show()

    runoff = np.mean(runoff[:,:365], axis=0)

    return runoff



project_folder = '/Users/mike/Documents/Research/Projects/Chukchi Sea'

era_xx_folder = '/Users/mike/Documents/Research/Projects/Ocean_Modeling/ECCO/LLC270_Files/era_xx'

print('Reading model grid...')
XC, YC, Depth = read_model_grid(project_folder)
points = reproject_points(np.column_stack((XC.flatten(), YC.flatten())), 4326, 32602)
X = points[:, 0].reshape(XC.shape)
Y = points[:, 1].reshape(YC.shape)

year = 2000

print('Reading precipitation data...')
precip_lon, precip_lat, precip = read_precip_data(era_xx_folder, year)

print('Subsetting precipitation data...')
# subset the grid to the region just around the XC and YC points
precip_lon_indices = np.logical_and(precip_lon>=np.min(XC)-1, precip_lon<=np.max(XC)+1)
precip_lat_indices = np.logical_and(precip_lat>=np.min(YC)-1, precip_lat<=np.max(YC)+1)
precip_lon = precip_lon[precip_lon_indices]
precip_lat = precip_lat[precip_lat_indices]
precip_2023 = precip[:, precip_lat_indices, :][:, :, precip_lon_indices]

precip_Lon, precip_Lat = np.meshgrid(precip_lon, precip_lat)
precip_points = reproject_points(np.column_stack((precip_Lon.flatten(), precip_Lat.flatten())), 4326, 32602)
precip_X = precip_points[:, 0].reshape(precip_Lon.shape)
precip_Y = precip_points[:, 1].reshape(precip_Lat.shape)

print('Interpolating year precipitation data to model grid...')
precip_climatology_on_domain = read_precip_to_model_grid(X, Y, precip_X, precip_Y, precip, testing=False)










