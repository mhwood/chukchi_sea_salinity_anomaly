
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4
from scipy.interpolate import griddata
from pyproj import Transformer
import cmocean.cm as cm
from scipy.io import loadmat
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon

def great_circle_distance(lon_ref, lat_ref, Lon, Lat):
    earth_radius = 6371000
    lon_ref_radians = np.radians(lon_ref)
    lat_ref_radians = np.radians(lat_ref)
    lons_radians = np.radians(Lon)
    lats_radians = np.radians(Lat)
    lat_diff = lats_radians - lat_ref_radians
    lon_diff = lons_radians - lon_ref_radians
    d = np.sin(lat_diff * 0.5) ** 2 + np.cos(lat_ref_radians) * np.cos(lats_radians) * np.sin(lon_diff * 0.5) ** 2
    h = 2 * earth_radius * np.arcsin(np.sqrt(d))
    return(h)

def read_grid_geometry_from_nc(project_dir):
    file_path = os.path.join(project_dir, 'Data','Model', 'Chukchi_Sea_grid.nc')
    ds = nc4.Dataset(file_path)
    XC = ds.variables['XC'][:,:]
    YC = ds.variables['YC'][:,:]
    Depth = ds.variables['Depth'][:,:]
    ds.close()
    return XC, YC, Depth

def read_gridded_observation_transects_from_mat(project_dir, transect):

    if transect=='Bering':
        file_name = 'section_bs_2023.mat'
    else:
        file_name = 'section_'+transect.lower()+'_2023.mat'

    # read the .mat file
    file_path = os.path.join(project_dir, 'Data', 'Observations','gridded_sections_2023', file_name)

    data = loadmat(file_path)

    # read in the X variable
    obs_X = data['X']
    obs_Y = data['Y']

    theta = data['the'][0][5] #['oct2023']
    salt = data['sal'][0][0]
    time = data['tim'][0][0]
    depth = data['bot'][0][0]
    # print(data.keys())
    # print(np.shape(theta))
    # print(type(theta))

def read_gridded_observation_transects_from_csv(project_dir, transect, var_name):

    if transect=='Bering':
        file_name = 'bs_oct2023_'+var_name.lower()+'.csv'
    else:
        file_name = transect.lower()+'_oct2023_'+var_name.lower()+'.csv'

    # read the .csv file
    file_path = os.path.join(project_dir, 'Data', 'Observations','gridded_sections_2023', file_name)
    data = np.genfromtxt(file_path, delimiter=',')

    # C = plt.pcolormesh(data)
    # plt.colorbar(C)
    # plt.show()

    return data

def dbo_mask_points(section):
    if section=='Bering':
        points =  np.array([[ 191.849 , 65.61883333333333 ],
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
                            [ 190.35783333333333 , 65.98166666666667 ]])
    if section=='DBO3':
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
    if section=='DBO5':
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
    return(points)

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

def series_to_N_points(series,N):
    #find the total length of the series
    totalDistance=0
    for s in range(len(series[:,0])-1):
        totalDistance+=((series[s,0]-series[s+1,0])**2+(series[s,1]-series[s+1,1])**2)**0.5
    intervalDistance=totalDistance/(N-1)

    #make the list of points
    newSeries=series[0,:]
    currentS = 0
    currentPoint1=series[currentS,:]
    currentPoint2=series[currentS+1,:]
    for p in range(N-2):
        distanceAccrued = 0
        while distanceAccrued<intervalDistance:
            currentLineDistance=((currentPoint1[0]-currentPoint2[0])**2+(currentPoint1[1]-currentPoint2[1])**2)**0.5
            if currentLineDistance<intervalDistance-distanceAccrued:
                distanceAccrued+=currentLineDistance
                currentS+=1
                currentPoint1 = series[currentS, :]
                currentPoint2 = series[currentS + 1, :]
            else:
                distance=intervalDistance-distanceAccrued
                newX=currentPoint1[0]+(distance/currentLineDistance)*(currentPoint2[0]-currentPoint1[0])
                newY = currentPoint1[1] + (distance / currentLineDistance) * (currentPoint2[1] - currentPoint1[1])
                distanceAccrued=intervalDistance+1
                newSeries=np.vstack([newSeries,np.array([newX,newY])])
                currentPoint1=np.array([newX,newY])
    newSeries = np.vstack([newSeries, series[-1,:]])
    return(newSeries)

def create_transect_line(transect_points_32602, X, Y, Depth):

    resolution = 100 # meters

    x = transect_points_32602[:, 0]
    y = transect_points_32602[:, 1]

    # Create a grid of points along the transect line
    num_points = int(np.ceil(np.linalg.norm(transect_points_32602[-1] - transect_points_32602[0]) / resolution))

    transect_points_32602 = series_to_N_points(transect_points_32602, num_points)
    x_line = transect_points_32602[:, 0]
    y_line = transect_points_32602[:, 1]

    transect_depth = griddata((X, Y), Depth.flatten(), (x_line, y_line), method='linear')

    transect_line = np.column_stack((x_line, y_line, transect_depth))
    return transect_line

def read_transect_profile_means_from_dv(config_dir, var_name, transect, date_str, transect_line_32602):
    file_path = os.path.join(config_dir, 'results_control','dv', transect, var_name, f'{var_name}_{date_str}.nc')
    ds = nc4.Dataset(file_path)
    depths = ds.variables['depths'][:]
    longitude = ds.variables['longitude'][:]
    latitude = ds.variables['latitude'][:]
    var_grid = ds.variables[var_name][:,:, :]
    ds.close()

    # print(np.min(var_grid), np.max(var_grid))

    var_grid = np.mean(var_grid, axis=0)  # Average over time
    # print(np.shape(var_grid))

    # Reproject the transect points to match the grid
    points = np.array([longitude.flatten(), latitude.flatten()]).T
    points_32602 = reproject_polygon(points, 4326, 32602)

    profile_means = np.zeros((len(depths), len(transect_line_32602), ))

    # plt.plot(points_32602[:, 0], points_32602[:, 1], 'r.', label='Model Grid Points')
    # plt.plot(transect_line_32602[:, 0], transect_line_32602[:, 1], 'k.', label='Transect Points')
    # plt.show()

    # iterate each depth level in the var grid if there are any nonzero points
    for i, depth in enumerate(depths):
        points_subset = np.copy(points_32602)
        values = var_grid[i, :].flatten()
        nonzero_indices = values!=0

        if np.sum(nonzero_indices) > 4:
            points_subset = points_subset[nonzero_indices, :]
            values = values[nonzero_indices]
            var_values = griddata(points_subset, values,
                                  (transect_line_32602[:,0],transect_line_32602[:,1]), method='linear')
            # if i==0:
            #     plt.plot(var_values)
            #     plt.show()

            profile_means[i, :] = var_values
        else:
            profile_means[i, :] = np.nan

    return(depths, profile_means)


def read_transect_profile_means_from_daily_nc(config_dir, var_name, transect, date_str, transect_points, transect_line_32602):
    var_name = var_name[:1]+var_name.lower()[1:]
    file_path = os.path.join(config_dir, 'results_control','daily_mean', var_name, f'{var_name}_{date_str}.nc')
    ds = nc4.Dataset(file_path)
    depths = ds.variables['depths'][:]
    longitude = ds.variables['longitude'][:]
    latitude = ds.variables['latitude'][:]
    var_grid = ds.variables[var_name][:,: :,:]
    ds.close()

    # sanity plot a profile
    # # find closest point to the first transect point
    # dist = great_circle_distance(transect_points[-1, 0], transect_points[-1, 1], longitude, latitude)
    # row, col = np.unravel_index(np.argmin(dist), dist.shape)
    # print(f'Transect start point: lon={transect_points[-1, 0]}, lat={transect_points[-1, 1]}')
    # print(f'Closest point to transect start: row={row}, col={col}, lon={longitude[row, col]}, lat={latitude[row, col]}')
    #
    # profile = np.column_stack([depths, var_grid[17, :, row, col]])
    # profile = profile[profile[:, 1]!=0, :]  # Remove NaN values
    # plt.plot(profile[:,1], profile[:,0])
    # plt.ylim([70,0])
    # plt.show()

    # Average over time
    var_grid = np.mean(var_grid, axis=0)

    # Reproject the transect points to match the grid
    points = np.array([longitude.flatten(), latitude.flatten()]).T
    points_32602 = reproject_polygon(points, 4326, 32602)
    X = points_32602[:, 0].reshape(np.shape(longitude))
    Y = points_32602[:, 1].reshape(np.shape(longitude))

    # subset domain to just the locations around the transect line
    min_x_index = np.argmin(np.abs(X[0, :] - np.min(transect_line_32602[:, 0])))
    max_x_index = np.argmin(np.abs(X[0, :] - np.max(transect_line_32602[:, 0])))
    min_y_index = np.argmin(np.abs(Y[:, 0] - np.min(transect_line_32602[:, 1])))
    max_y_index = np.argmin(np.abs(Y[:, 0] - np.max(transect_line_32602[:, 1])))
    print(min_x_index, max_x_index, min_y_index, max_y_index)
    var_grid = var_grid[:, min_y_index:max_y_index+1, min_x_index:max_x_index+1]
    X = X[min_y_index:max_y_index+1, min_x_index:max_x_index+1]
    Y = Y[min_y_index:max_y_index+1, min_x_index:max_x_index+1]
    points_32602 = np.column_stack((X.flatten(), Y.flatten()))

    # empty grid to fill in
    profile_means = np.zeros((len(depths), len(transect_line_32602), ))

    plt.plot(points_32602[:, 0], points_32602[:, 1], 'r.', label='Model Grid Points')
    plt.plot(transect_line_32602[:, 0], transect_line_32602[:, 1], 'k.', label='Transect Points')
    plt.show()

    # Average over time

    # iterate each depth level in the var grid if there are any nonzero points
    for i, depth in enumerate(depths):
        points_subset = np.copy(points_32602)
        values = var_grid[i, :, :].flatten()
        nonzero_indices = values!=0

        if np.sum(nonzero_indices) > 4:
            points_subset = points_subset[nonzero_indices, :]
            values = values[nonzero_indices]
            var_values = griddata(points_subset, values,
                                  (transect_line_32602[:,0],transect_line_32602[:,1]), method='linear')
            if i==0:
                plt.plot(var_values)
                plt.show()

            profile_means[i, :] = var_values
        else:
            profile_means[i, :] = np.nan

    return(depths, profile_means)


def plot_transect_profile_means(project_dir, transect, date_str, transect_line_32602, transect_distance,
                            obs_theta_grid_profile, obs_salt_grid_profile,
                            model_depth, model_theta_grid_profile, model_salt_grid_profile):

    if transect=='Bering':
        ymin = 65
        obs_distance = np.array([7.500, 10, 12.50, 15, 17.50, 20, 22.50, 25, 27.50, 30, 32.50, 35, 37.50, 40, 42.50])
        obs_depths = 2.5*np.arange(np.shape(obs_theta_grid_profile)[0])
    if transect=='DBO3':
        ymin = 65
        obs_distance = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
        obs_depths = 2.5*np.arange(np.shape(obs_theta_grid_profile)[0])

    cmap_theta = cm.thermal
    cmap_salt = cm.haline

    vmin_theta = -2.0
    vmax_theta = 8.0
    vmin_salt = 27.0
    vmax_salt = 33.0

    plot_width = 10
    plot_height = 8

    bathy_polygon_top = np.column_stack([transect_distance, transect_line_32602[:,2]])
    bathy_polygon_bottom = np.column_stack([transect_distance, ymin*np.ones_like(transect_distance)])
    bathy_polygon = np.vstack([bathy_polygon_top, bathy_polygon_bottom[::-1, :]])


    # Plotting
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2*plot_height+1, 2*plot_width+4, figure=fig, left=0.08, right=0.92, top=0.95, bottom=0.08)

    plt.style.use('dark_background')

    ########################################################################################
    # upper left plot - plot observed temperature

    ax1 = fig.add_subplot(gs[:plot_height, :plot_width])
    C = ax1.pcolormesh(obs_distance, obs_depths,
                       obs_theta_grid_profile, shading='auto',
                       cmap=cmap_theta, vmin=vmin_theta, vmax=vmax_theta)
    #plt.colorbar(C, label=var_name)
    #plt.plot(transect_line[:, 0], transect_line[:, 2], 'k-', label='Transect Depth')
    # ax1.set_xlabel('Longitude (EPSG:32602)')
    ax1.set_ylabel('Estimated Depth (m)')
    ax1.invert_yaxis()
    ax1.set_title(f'Observed {transect} Transect Profile')
    ax1.set_ylim([ymin, 0])

    ########################################################################################
    # upper right plot - plot modeled temperature
    ax2 = fig.add_subplot(gs[:plot_height, plot_width+2:2*plot_width+2])
    C = ax2.pcolormesh(transect_distance, model_depth,
                       model_theta_grid_profile, shading='auto',
                       cmap=cmap_theta, vmin=vmin_theta, vmax=vmax_theta)
    # ax2.add_patch(Polygon(bathy_polygon, closed=True, color='silver', label='Bathymetry'))
    #plt.colorbar(C, label=var_name)
    # plt.plot(transect_line[:, 0], transect_line[:, 2], 'k-', label='Transect Depth')
    ax2.set_ylim([ymin, 0])
    # ax2.set_xlabel('Longitude (EPSG:32602)')
    ax2.set_ylabel('Model Depth (m)')
    ax2.set_title(f'Modeled {transect} Transect Profile')

    ##################################################################
    # Make a manual colorbar for the temperature plots
    cbar_ax1 = fig.add_subplot(gs[0:plot_height, -1:])
    cbar1 = fig.colorbar(C, cax=cbar_ax1, orientation='vertical')
    cbar1.set_label('Temperature (°C)', fontsize=12)

    ########################################################################################
    # lower left plot - plot observed salinity

    ax3 = fig.add_subplot(gs[plot_height+1:2*plot_height+1, :plot_width])
    C = ax3.pcolormesh(obs_distance, obs_depths,
                       obs_salt_grid_profile, shading='auto',
                          cmap=cmap_salt, vmin=vmin_salt, vmax=vmax_salt)
    #plt.colorbar(C, label=var_name)
    #plt.plot(transect_line[:, 0], transect_line[:, 2], 'k-', label='Transect Depth')
    ax3.set_xlabel('Distance Along Transect (km)')
    ax3.set_ylabel('Estimated Depth (m)')
    ax3.invert_yaxis()
    ax3.set_ylim([ymin, 0])

    ########################################################################################
    # lower right plot - plot modeled salinity
    ax4 = fig.add_subplot(gs[plot_height+1:2*plot_height+1, plot_width+2:2*plot_width+2])
    C = ax4.pcolormesh(transect_distance, model_depth,
                       model_salt_grid_profile, shading='auto',
                       cmap=cmap_salt, vmin=vmin_salt, vmax=vmax_salt)
    #plt.colorbar(C, label=var_name)
    # plt.plot(transect_line[:, 0], transect_line[:, 2], 'k-', label='Transect Depth')
    ax4.set_ylim([ymin, 0])
    ax4.set_xlabel('Distance Along Transect (EPSG:32602)')
    ax4.set_ylabel('Model Depth (m)')

    ##################################################################
    # Make a manual colorbar for the salinity plots

    cbar_ax2 = fig.add_subplot(gs[plot_height+1:2*plot_height+1, -1:])
    cbar2 = fig.colorbar(C, cax=cbar_ax2, orientation='vertical')
    cbar2.set_label('Salinity (psu)', fontsize=12)


    output_file = os.path.join(project_dir, 'Figures', 'Model', f'{transect}_transect_profile_{date_str}.png')
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

project_dir = '/Users/mike/Documents/Research/Projects/Chukchi Sea'

config_dir = '/Volumes/kullorsuaq/Research/Ocean_Modeling/Projects/Chukchi_Sea'


# transect = 'DBO3'
transect = 'Bering'
date_str = '202310'

print(f'Reading gridded observation transects for {transect}...')
obs_theta_grid_profile = read_gridded_observation_transects_from_csv(project_dir, transect, var_name="THETA")
obs_salt_grid_profile = read_gridded_observation_transects_from_csv(project_dir, transect, var_name="SALT")

print('obs theta shape:', np.shape(obs_theta_grid_profile))
print('obs salt shape:', np.shape(obs_salt_grid_profile))

print('Reading transect points and reprojecting them to EPSG:32602...')
transect_points = dbo_mask_points(transect)
transect_points_32602_coarse = reproject_polygon(transect_points, 4326, 32602)

XC, YC, Depth = read_grid_geometry_from_nc(project_dir)
model_points = np.array([XC.flatten(), YC.flatten()])
model_points_32602 = reproject_polygon(model_points.T, 4326, 32602)
X = model_points_32602[:, 0]
Y = model_points_32602[:, 1]

transect_line_32602 = create_transect_line(transect_points_32602_coarse, X, Y, Depth)
transect_distance = np.zeros((np.shape(transect_line_32602)[0],))
for i in range(1, np.shape(transect_line_32602)[0]):
    transect_distance[i] = transect_distance[i-1] + np.sqrt((transect_line_32602[i, 0] - transect_line_32602[i-1, 0])**2 +
                                                            (transect_line_32602[i, 1] - transect_line_32602[i-1, 1])**2)
transect_distance *= 1e-3 # Convert to kilometers
# plt.plot(transect_distance)
# plt.show()

# plt.subplot(1,2,1)
# plt.plot(transect_points[:, 0], transect_points[:, 1], 'ko', label='Transect Points')
# plt.subplot(1,2,2)
# plt.plot(transect_points_32602_coarse[:, 0], transect_points_32602_coarse[:, 1], 'ko', label='Transect Points')
# plt.plot(transect_line_32602[:, 0], transect_line_32602[:, 1], 'r.', label='Transect Line')
# plt.show()

print('Reading transect profile means...')
# model_depth, model_theta_grid_profile = read_transect_profile_means_from_dv(config_dir, "THETA", transect, date_str, transect_line_32602)
# _, model_salt_grid_profile = read_transect_profile_means_from_dv(config_dir, "SALT", transect, date_str, transect_line_32602)
model_depth, model_theta_grid_profile = read_transect_profile_means_from_daily_nc(config_dir, "THETA", transect, date_str, transect_points, transect_line_32602)
_, model_salt_grid_profile = read_transect_profile_means_from_daily_nc(config_dir, "SALT", transect, date_str, transect_points, transect_line_32602)

# plt.pcolormesh(model_theta_grid_profile, shading='auto')
# plt.show()

print(np.nanmin(model_theta_grid_profile), np.nanmax(model_theta_grid_profile))
print(np.nanmin(model_salt_grid_profile), np.nanmax(model_salt_grid_profile))

print('Plotting transect profile means...')
plot_transect_profile_means(project_dir, transect, date_str, transect_line_32602, transect_distance,
                            obs_theta_grid_profile, obs_salt_grid_profile,
                            model_depth, model_theta_grid_profile, model_salt_grid_profile)


