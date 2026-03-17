
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

def read_modeled_DBO_section(project_folder, var_name, experiment, dbo_section):

    #'DBO Sections',
    dbo_file_name = os.path.join(project_folder,'Data', 'Model','DBO Sections',
                                 var_name + '_DBO_Sections_' + experiment + '.nc')
    ds = nc4.Dataset(dbo_file_name)

    grp = ds.groups[dbo_section]
    dec_yrs = grp.variables['dec_yrs'][:]
    Z = grp.variables['depth'][:]
    var_grid = grp.variables[var_name][:,:,:]

    ds.close()
    return dec_yrs, Z, var_grid

def plot_monthly_mean_DBO_fields(project_folder, year, transect, var_name, dec_yrs, Z, var_grid,
                                 coordinates_4326, coordinates_32602):

    plot_width = 9
    plot_height = 9

    distance_along_transect = np.zeros(coordinates_32602.shape[0])
    for i in range(1,coordinates_32602.shape[0]):
        distance_along_transect[i] = distance_along_transect[i-1] + np.sqrt(
            (coordinates_32602[i,0]-coordinates_32602[i-1,0])**2 +
            (coordinates_32602[i,1]-coordinates_32602[i-1,1])**2)
    distance_along_transect /= 1000

    metadata_dict = {'Theta':['Potential Temperature', '°C', cm.thermal, -2, 10],
                     'Salt':['Salinity', 'PSU', cm.haline, 25, 33]}

    if transect=='DBO3':
        zmin=0
        zmax=60
    elif transect=='BS':
        zmin=0
        zmax=50
    else:
        zmin=0
        zmax=100

    fig = plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    gs = GridSpec(2*plot_height+1, 2*plot_width+2, left=0.07, right=0.9, top=0.90, bottom=0.1,
                  wspace=0.3, hspace=0.3)

    for month in range(6,10):
        if month==6:
            ax = fig.add_subplot(gs[:plot_width, :plot_width])
        elif month==7:
            ax = fig.add_subplot(gs[:plot_height, plot_width:2 * plot_width])
        elif month==8:
            ax = fig.add_subplot(gs[-plot_height:, :plot_width])
        elif month==9:
            ax = fig.add_subplot(gs[-plot_height:, plot_width:2*plot_width])

        # Find indices for the month
        start_dec_yr = YMD_to_DecYr(year, month, 1)
        if month < 12:
            end_dec_yr = YMD_to_DecYr(year, month+1, 1)
        else:
            end_dec_yr = YMD_to_DecYr(year+1, 1, 1)
        month_indices = np.where((dec_yrs >= start_dec_yr) & (dec_yrs < end_dec_yr))[0]
        # Compute monthly mean
        monthly_mean = np.nanmean(var_grid[month_indices, :, :], axis=0)

        #testing
        # monthly_mean = np.nanmean(var_grid[:, :, :], axis=0)

        # Plot
        plt.pcolormesh(distance_along_transect, Z, monthly_mean, shading='auto', cmap = metadata_dict[var_name][2],
                          vmin=metadata_dict[var_name][3], vmax=metadata_dict[var_name][4])
        plt.contour(distance_along_transect, Z, monthly_mean,
                    levels=np.linspace(metadata_dict[var_name][3], metadata_dict[var_name][4], 11),
                    colors='k', linewidths=0.5)
        ax.set_ylim([zmax, zmin])
        ax.set_title('Month: ' + str(month))
        if month in [7,9]:
            ax.set_yticks([])
        else:
            ax.set_ylabel('Depth (m)')
        if month in [6,7]:
            ax.set_xticks([])
        else:
            ax.set_xlabel('Distance From Alaska (km)')
        ax.invert_xaxis()

    ax = fig.add_subplot(gs[3:-3, -1])
    x = np.array([0,1])
    y = np.linspace(metadata_dict[var_name][3], metadata_dict[var_name][4], 100)
    X, Y = np.meshgrid(x, y)
    c = ax.pcolormesh(X, Y, Y, shading='auto', cmap=metadata_dict[var_name][2],
                      vmin=metadata_dict[var_name][3], vmax=metadata_dict[var_name][4])
    ax.set_xticks([])
    ax.set_ylabel(metadata_dict[var_name][0]+' (' + metadata_dict[var_name][1] + ')')
    # move yaxis ticks and lebels to the right
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    plt.suptitle('Monthly Mean ' + metadata_dict[var_name][0] + ' ' + transect + ' '+ str(year), fontsize=12)

    plt.savefig(os.path.join(project_folder, 'Figures','Model', f'Monthly_Mean_{var_name}_{transect}_{year}.png'))
    plt.close(fig)



project_folder = '/Users/mike/Documents/Research/Projects/Chukchi Sea/'

dbo_point_set = {}
dbo_point_set['BS'] = reproject_points(get_DBO_coordinates('BS'), 4326, 32602)
dbo_point_set['DBO3'] = reproject_points(get_DBO_coordinates('DBO3'), 4326, 32602)
dbo_point_set['DBO5'] = reproject_points(get_DBO_coordinates('DBO5'), 4326, 32602)

year = 2023
var_name = 'Vvel'

anomaly = True

for var_name in ['Theta', 'Salt']:
    for transect in ['BS']:#,'DBO3','DBO5']:

        dec_yrs, Z, var_grid = read_modeled_DBO_section(project_folder, var_name, 'daily_hydrology', transect)

        coordinates_4326 = get_DBO_coordinates(transect)
        coordinates_32602 = reproject_points(get_DBO_coordinates(transect), 4326, 32602)

        plot_monthly_mean_DBO_fields(project_folder, year, transect, var_name, dec_yrs, Z, var_grid,
                                     coordinates_4326, coordinates_32602)






