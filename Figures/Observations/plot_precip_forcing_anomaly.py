
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

def plot_summer_precip_anomalies(project_folder, X, Y, Depth,
                                 precip_on_domain_2023, precip_on_domain_2024, precip_climatology_on_domain):

    fig = plt.figure(figsize=(12, 9))

    plot_width = 8

    gs = GridSpec(3, plot_width*3+2, left=0.08, right=0.92, bottom=0.05, top=0.95, hspace=0.05)

    vmin = -8
    vmax = 8

    #####################################################################################
    # July precipitation anomaly
    ax = plt.subplot(gs[0, :plot_width])
    july_indices = np.arange(180*4, 212*4)  # July is days 181-212 in a non-leap year
    precip_anomaly_july = np.mean(precip_on_domain_2023[july_indices, :, :], axis=0) - \
                            np.mean(precip_climatology_on_domain[july_indices, :, :], axis=0)
    plt.pcolormesh(X, Y, precip_anomaly_july,cmap='seismic', vmin=vmin, vmax=vmax)
    plt.contour(X, Y, Depth, levels=[0], colors='k', linewidths=0.5)
    plt.title('July Mean')
    plt.ylabel('2023')
    ax.set_xticks([])
    ax.set_yticks([])

    #####################################################################################
    # August precipitation anomaly
    ax = plt.subplot(gs[0, plot_width:2*plot_width])
    august_indices = np.arange(212*4, 243*4)  # August is days 213-243 in a non-leap year
    precip_anomaly_august = np.mean(precip_on_domain_2023[august_indices, :, :], axis=0) - \
                np.mean(precip_climatology_on_domain[august_indices, :, :], axis=0)
    plt.pcolormesh(X, Y, precip_anomaly_august, cmap='seismic', vmin=vmin, vmax=vmax)
    plt.contour(X, Y, Depth, levels=[0], colors='k', linewidths=0.5)
    plt.title('August Mean')
    ax.set_xticks([])
    ax.set_yticks([])

    #####################################################################################
    # September precipitation anomaly
    ax = plt.subplot(gs[0, 2*plot_width:3*plot_width])
    september_indices = np.arange(243*4, 273*4)  # September is days 244-273 in a non-leap year
    precip_anomaly_september = np.mean(precip_on_domain_2023[september_indices, :, :], axis=0) - \
                np.mean(precip_climatology_on_domain[september_indices, :, :], axis=0)
    plt.pcolormesh(X, Y, precip_anomaly_september, cmap='seismic', vmin=vmin, vmax=vmax)
    plt.contour(X, Y, Depth, levels=[0], colors='k', linewidths=0.5)
    plt.title('September Mean')
    ax.set_xticks([])
    ax.set_yticks([])

    #####################################################################################
    # July precipitation anomaly
    ax = plt.subplot(gs[1, :plot_width])
    july_indices = np.arange(180 * 4, 212 * 4)  # July is days 181-212 in a non-leap year
    precip_anomaly_july = np.mean(precip_on_domain_2024[july_indices, :, :], axis=0) - \
                          np.mean(precip_climatology_on_domain[july_indices, :, :], axis=0)
    plt.pcolormesh(X, Y, precip_anomaly_july, cmap='seismic', vmin=vmin, vmax=vmax)
    plt.contour(X, Y, Depth, levels=[0], colors='k', linewidths=0.5)
    plt.ylabel('2024')
    ax.set_xticks([])
    ax.set_yticks([])

    #####################################################################################
    # August precipitation anomaly
    ax = plt.subplot(gs[1, plot_width:2 * plot_width])
    august_indices = np.arange(212 * 4, 243 * 4)  # August is days 213-243 in a non-leap year
    precip_anomaly_august = np.mean(precip_on_domain_2024[august_indices, :, :], axis=0) - \
                            np.mean(precip_climatology_on_domain[august_indices, :, :], axis=0)
    plt.pcolormesh(X, Y, precip_anomaly_august, cmap='seismic', vmin=vmin, vmax=vmax)
    plt.contour(X, Y, Depth, levels=[0], colors='k', linewidths=0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    #####################################################################################
    # September precipitation anomaly
    ax = plt.subplot(gs[1, 2 * plot_width:3 * plot_width])
    september_indices = np.arange(243 * 4, 273 * 4)  # September is days 244-273 in a non-leap year
    precip_anomaly_september = np.mean(precip_on_domain_2024[september_indices, :, :], axis=0) - \
                               np.mean(precip_climatology_on_domain[september_indices, :, :], axis=0)
    plt.pcolormesh(X, Y, precip_anomaly_september, cmap='seismic', vmin=vmin, vmax=vmax)
    plt.contour(X, Y, Depth, levels=[0], colors='k', linewidths=0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    #####################################################################################
    # Manual colorbar
    ax = plt.subplot(gs[:2, -1:])
    x = np.array([0,1])
    y = np.linspace(vmin, vmax, 100)
    X,Y = np.meshgrid(x, y)
    C = plt.pcolormesh(X, Y, Y, cmap='seismic', vmin=vmin, vmax=vmax)
    plt.ylabel('Precipitation Anomaly Relative to 1992-2024 Mean (mm/day)')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_xticks([])

    #####################################################################################
    # Timeseries

    ax = plt.subplot(gs[2, :])
    days = np.arange(1, 400, 0.25)  # Days of the year

    precip_timeseries_2023 = np.mean(precip_on_domain_2023, axis=(1, 2))  # Mean precipitation for each day
    days_2023 = days[:len(precip_timeseries_2023)]
    plt.plot(days_2023, precip_timeseries_2023, label='2023', color='blue')

    precip_timeseries_2024 = np.mean(precip_on_domain_2024, axis=(1, 2))  # Mean precipitation for each day
    days_2024 = days[:len(precip_timeseries_2024)]
    plt.plot(days_2024, precip_timeseries_2024, label='2024', color='green')

    precip_timeseries_climatology = np.mean(precip_climatology_on_domain, axis=(1, 2))  # Mean climatology for each day
    days_climatology = days[:len(precip_timeseries_climatology)]
    plt.plot(days_climatology, precip_timeseries_climatology, label='Climatology', color='orange')

    plt.legend()
    plt.xlabel('Day of the Year')
    plt.ylabel('Mean Precipitation in Model Domain\n (mm/day)')

    plt.savefig(os.path.join(project_folder, 'Figures', 'Model', 'Chukchi_Sea_Precipitation.png'),dpi=300)
    plt.close(fig)

testing = False

project_folder = '/Users/mike/Documents/Research/Projects/Chukchi Sea'

era_xx_folder = '/Users/mike/Documents/Research/Projects/Ocean_Modeling/ECCO/LLC270_Files/era_xx'

print('Reading model grid...')
XC, YC, Depth = read_model_grid(project_folder)
points = reproject_points(np.column_stack((XC.flatten(), YC.flatten())), 4326, 32602)
X = points[:, 0].reshape(XC.shape)
Y = points[:, 1].reshape(YC.shape)

print('Reading precipitation data...')
precip_lon, precip_lat, precip_2023 = read_precip_data(era_xx_folder, 2023)
_, _, precip_2024 = read_precip_data(era_xx_folder, 2024)
_, _, precip_climatology = read_precip_data(era_xx_folder, 'climatology')

print('Subsetting precipitation data...')
# subset the grid to the region just around the XC and YC points
precip_lon_indices = np.logical_and(precip_lon>=np.min(XC)-1, precip_lon<=np.max(XC)+1)
precip_lat_indices = np.logical_and(precip_lat>=np.min(YC)-1, precip_lat<=np.max(YC)+1)
precip_lon = precip_lon[precip_lon_indices]
precip_lat = precip_lat[precip_lat_indices]
precip_2023 = precip_2023[:, precip_lat_indices, :][:, :, precip_lon_indices]
precip_2024 = precip_2024[:, precip_lat_indices, :][:, :, precip_lon_indices]
precip_climatology = precip_climatology[:, precip_lat_indices, :][:, :, precip_lon_indices]
precip_Lon, precip_Lat = np.meshgrid(precip_lon, precip_lat)
precip_points = reproject_points(np.column_stack((precip_Lon.flatten(), precip_Lat.flatten())), 4326, 32602)
precip_X = precip_points[:, 0].reshape(precip_Lon.shape)
precip_Y = precip_points[:, 1].reshape(precip_Lat.shape)

print('Interpolating year precipitation data to model grid...')
precip_climatology_on_domain = read_precip_to_model_grid(X, Y, precip_X, precip_Y, precip_climatology, testing)

print('Interpolating climatology precipitation data to model grid in 2023...')
precip_on_domain_2023 = read_precip_to_model_grid(X, Y, precip_X, precip_Y, precip_2023, testing)

print('Interpolating climatology precipitation data to model grid in 2024...')
precip_on_domain_2024 = read_precip_to_model_grid(X, Y, precip_X, precip_Y, precip_2024, testing)

print(np.min(precip_on_domain_2023), np.max(precip_on_domain_2023))
print(np.min(precip_on_domain_2024), np.max(precip_on_domain_2024))
print(np.min(precip_climatology_on_domain), np.max(precip_climatology_on_domain))

print('Plotting summer precipitation anomalies...')
plot_summer_precip_anomalies(project_folder, X, Y, Depth,
                             precip_on_domain_2023, precip_on_domain_2024, precip_climatology_on_domain)






