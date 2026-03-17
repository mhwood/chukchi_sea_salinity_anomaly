

import os
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc4
from pyproj import Transformer
import cmocean.cm as cm
from matplotlib.patches import Rectangle
from datetime import timedelta, datetime
# import Gridspec
from matplotlib.gridspec import GridSpec
# ignore UserWarning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def date_to_iter_number(date,seconds_per_iter,start_year=1992):
    total_seconds = (date-datetime(start_year,1,1)).total_seconds()
    iter_number = total_seconds/seconds_per_iter
    return(iter_number)

def make_iter_number_list(year, var_name):

    iter_numbers = []
    date_strs = []

    for month in range(1, 13):
        if month in [1, 3, 5, 7, 8, 10, 12]:
            days_in_month = 31
        elif month in [4, 6, 9, 11]:
            days_in_month = 30
        elif month == 2:
            if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                days_in_month = 29
            else:
                days_in_month = 28
        if year==2024 and month==12:
            days_in_month=29
        for day in range(1, days_in_month + 1):
            date = datetime(year, month, day)
            if 'SI' in var_name:
                date = date + timedelta(hours=12)
            iter_number = int(date_to_iter_number(date, 30))
            iter_numbers.append(iter_number)
            date_strs.append(date.strftime('%Y%m%d'))
    return iter_numbers, date_strs

def read_MODIS_imagery(project_folder):

    ds = nc4.Dataset(os.path.join(project_folder, 'Data','Imagery', 'Chukchi_Sea_MODIS_20220720_32602.nc'))
    x = ds.variables['x'][:]
    y = ds.variables['y'][:]
    band_1 = ds.variables['band_1'][:,:]
    band_3 = ds.variables['band_3'][:,:]
    band_4 = ds.variables['band_4'][:,:]
    ds.close()

    # flipud all the bands
    band_1 = np.flipud(band_1)
    band_3 = np.flipud(band_3)
    band_4 = np.flipud(band_4)

    # normalize the bands
    band_1 = band_1/1.6
    band_3 = band_3/1.6
    band_4 = band_4/1.6

    band_1[band_1 < 0] = 0
    band_3[band_3 < 0] = 0
    band_4[band_4 < 0] = 0

    img = np.concatenate((band_1[:,:,np.newaxis], band_4[:,:,np.newaxis], band_3[:,:,np.newaxis]), axis=2)

    # brightness correction
    img = img / np.max(img)
    img = np.clip(img, 0, 1)


    #increase brightness
    img = img * 4
    img[img > 1] = 1

    img = (img * 255).astype(np.uint8)

    return(x,y, img)

def read_model_grid(project_folder):
    ds = nc4.Dataset(os.path.join(project_folder,'Data', 'Model', 'Chukchi_Sea_grid.nc'))
    XC = ds.variables['XC'][:,:]
    YC = ds.variables['YC'][:,:]
    Depth = ds.variables['Depth'][:,:]
    ds.close()
    return(XC, YC, Depth)

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

def read_model_field_from_mds(project_folder, var_name, iter_number):

    depth_level = 5

    if var_name=='SIheff':
        subset = 'SI_day_snap'
        file_name = subset + '.' + str(iter_number).zfill(10) + '.data'
        n_levels = 5
        lev = 1
    elif var_name=='Theta':
        subset = 'state_3D_day_mean'
        file_name = subset + '.' + str(iter_number).zfill(10) + '.data'
        n_levels = 90*2
        lev = depth_level
    elif var_name=='Salt':
        subset = 'state_3D_day_mean'
        file_name = subset + '.' + str(iter_number).zfill(10) + '.data'
        n_levels = 90*2
        lev = depth_level + 90

    data = np.fromfile(os.path.join(project_folder,'run','diags',subset,file_name), dtype='>f4')
    data = data.reshape((n_levels, 720, 960))
    data = data[lev,:,:]

    return data

def var_name_to_subset_and_filename(var_name, date_str):
    if var_name in ['Theta','Salt']:
        subset = 'daily_mean'
        file_name = var_name + '_' + date_str[:6] + '.nc'
    elif var_name in ['SIheff','SIarea']:
        subset = 'daily_snapshot'
        file_name = var_name + '_' + date_str[:6] + '.nc'
    return subset, file_name

def read_model_field_from_nc(config_dir, results_dir, subset, var_name, year, month):
    depth_level = 5

    if var_name in ['Theta','Salt']:
        subset = 'daily_mean'
        file_name = var_name + '_' + f'{year}{month:02d}' + '.nc'
        lev = depth_level
    elif var_name in ['SIheff','SIarea']:
        subset = 'daily_snapshot'
        file_name = var_name + '_' + f'{year}{month:02d}' + '.nc'
        lev = 0

    ds = nc4.Dataset(os.path.join(config_dir, results_dir, subset, var_name, file_name))
    if var_name in ['Theta', 'Salt']:
        data = ds.variables[var_name][:,lev,:,:]
    elif var_name in ['SIheff','SIarea']:
        data = ds.variables[var_name][:,:,:]
    ds.close()

    return data

def read_velocity_field(project_folder, file_name):

    depth_level = 5

    data = np.fromfile(os.path.join(project_folder,'Model','run','diags','vel_3D_day_mean',file_name), dtype='>f4')
    data = data.reshape((3*90, 720, 960))
    speed = np.sqrt(data[depth_level,:,:]**2 + data[depth_level+90,:,:]**2)
    U = data[depth_level,:,:]
    V = data[depth_level+90,:,:]

    return U, V, speed

def plot_panel(config_dir, results_dir, var_name, data, date_str, x, y, img, Depth):

    file_name = var_name+'_'+ date_str + '.png'
    plots_dir = results_dir.replace('results','plots')
    if var_name not in os.listdir(os.path.join(config_dir, plots_dir)):
        os.mkdir(os.path.join(config_dir, plots_dir, var_name))
    output_folder = os.path.join(config_dir, plots_dir,var_name)

    if file_name not in os.listdir(output_folder):
        print(f'            - Plotting {var_name} for {date_str} in '+plots_dir)

        day = int(date_str[6:8])
        # data = read_model_field_from_nc(config_dir, var_name, day, date_str)
        # data = read_model_field_from_mds(config_dir, var_name, iter_number)

        plot_metadata = {'SIheff': {'units': 'm', 'long_name': 'Sea Ice Thickness', 'cmap': cm.ice, 'vmin': 0, 'vmax': 2},
                         'SIarea': {'units': 'm$^2$/m$^2$', 'long_name': 'Sea Ice Concentration', 'cmap': cm.ice, 'vmin': 0,
                                    'vmax': 1},
                         'Theta': {'units': 'C', 'long_name': 'Sea Surface Temperature', 'cmap': cm.thermal, 'vmin': -2,
                                   'vmax': 10},
                         'Salt': {'units': 'psu', 'long_name': 'Salinity', 'cmap': cm.haline, 'vmin': 25, 'vmax': 35},
                         'UVEL': {'units': 'm/s', 'long_name': 'Eastward Velocity', 'cmap': 'viridis', 'vmin': -0.5,
                                  'vmax': 0.5},
                         'VVEL': {'units': 'm/s', 'long_name': 'Northward Velocity', 'cmap': 'viridis', 'vmin': -0.5,
                                  'vmax': 0.5},
                         'Speed': {'units': 'm/s', 'long_name': 'Speed', 'cmap': 'viridis', 'vmin': 0, 'vmax': 1}}


        fig = plt.figure(figsize=(10, 8))
        plt.style.use('dark_background')

        gs = GridSpec(17, 11, figure=fig, left = 0.02, right = 0.99, top = 0.91, bottom = 0.12)

        ax = fig.add_subplot(gs[:-2, :])

        plot_grid = np.ma.masked_where(Depth == 0, data)

        # add a white Rectangle
        rect = Rectangle((x.min(), y.min()), x.max() - x.min(), y.max() - y.min(), linewidth=1, edgecolor='w',
                         facecolor='white', zorder=0)
        plt.gca().add_patch(rect)

        plt.imshow(img, extent=(x.min(), x.max(), y.min(), y.max()), alpha=0.8, zorder=2)
        C = plt.imshow(plot_grid, extent=(X.min(), X.max(), Y.min(), Y.max()), cmap=plot_metadata[var_name]['cmap'],
                       origin='lower',
                       vmin=plot_metadata[var_name]['vmin'], vmax=plot_metadata[var_name]['vmax'], zorder=3)
        plt.colorbar(C, label=plot_metadata[var_name]['units'], orientation='vertical', pad=0.02, aspect=40,
                     shrink=0.8, ticks=np.linspace(plot_metadata[var_name]['vmin'], plot_metadata[var_name]['vmax'], 6))

        # add a white contour on the coastlines
        plt.contour(X, Y, Depth, levels=[0], colors='w', linewidths=0.5, zorder=4)

        # if var_name=='Speed':
        #     # add a quiver
        #     skip = 20
        #     U = U[::skip, ::skip]
        #     V = V[::skip, ::skip]
        #     U[U==0] = np.nan
        #     V[V==0] = np.nan
        #     plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], U, V, color='w', scale=10)

        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        if 'SI' in var_name:
            plt.title(plot_metadata[var_name]['long_name'],
                      fontsize=12)
        else:
            plt.title(plot_metadata[var_name]['long_name'] + ' (5m depth)',
                      fontsize=12)

        # add labels for Alaska and Russia with semitransparent bounding boxes
        ax.text(x.min() + 0.85 * (x.max() - x.min()), y.min() + 0.65 * (y.max() - y.min()), 'Alaska',fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'),
                ha='right', va='center', zorder=5)
        ax.text(x.min() + 0.2 * (x.max() - x.min()), y.min() + 0.6 * (y.max() - y.min()), 'Russia', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'),
                ha='left', va='top', zorder=5)

        # set the extent to remove 3% from edges
        plt.xlim(x.min() + 0.03 * (x.max() - x.min()), x.max() - 0.03 * (x.max() - x.min()))
        plt.ylim(y.min() + 0.03 * (y.max() - y.min()), y.max() - 0.03 * (y.max() - y.min()))

        ax3 = fig.add_subplot(gs[-1, 1:-2])
        date = datetime.strptime(date_str, '%Y%m%d')
        min_iter = date_to_iter_number(datetime(2023, 1, 1), 30)
        max_iter = date_to_iter_number(datetime(2025, 1, 1), 30)
        iter_number = date_to_iter_number(date, 30)
        # print(iter_number, min_iter, max_iter)
        if date.year==2023:
            width = (iter_number - min_iter) / (max_iter - min_iter)
        else:
            width = 2*(iter_number - min_iter) / (max_iter - min_iter)
        rect = Rectangle((2023, 0), width, 1, fc='silver', ec='white')
        ax3.add_patch(rect)

        ax3.set_ylim([0, 1])
        for i in range(2, 13):
            month_start_iter = date_to_iter_number(datetime(2023, i, 1), 30)
            x = 2023 + 2*(month_start_iter - min_iter) / (max_iter - min_iter)
            plt.plot([x,x], [0, 1], 'w-', linewidth=0.5)
        for i in range(1, 13):
            month_start_iter = date_to_iter_number(datetime(2024, i, 1), 30)
            x = 2023 + 2*(month_start_iter - min_iter) / (max_iter - min_iter)
            plt.plot([x,x], [0, 1], 'w-', linewidth=0.5)
        ax3.set_xticks(np.arange(2023 + 1 / 24, 2023 + 2, 1 / 12))
        ax3.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D',
                             'J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        ax3.set_yticks([])
        ax3.set_xlim([2023, 2025])


        ax3.annotate('', xy=(2023, -1.3), xytext=(2024, -1.3),  # draws an arrow from one set of coordinates to the other
                    arrowprops=dict(arrowstyle='<->', facecolor='white'),  # sets style of arrow and colour
                    annotation_clip=False)  # This enables the arrow to be outside of the plot

        ax3.annotate('2023', xy=(2023, -1.9), xytext=(2023.45, -1.9),  # Adds another annotation for the text that you want
                    annotation_clip=False, color='white', fontsize=12)

        ax3.annotate('', xy=(2024, -1.3), xytext=(2025, -1.3),
                     # draws an arrow from one set of coordinates to the other
                     arrowprops=dict(arrowstyle='<->', facecolor='white'),  # sets style of arrow and colour
                     annotation_clip=False)  # This enables the arrow to be outside of the plot

        ax3.annotate('2024', xy=(2024, -1.9), xytext=(2024.45, -1.9),
                     # Adds another annotation for the text that you want
                     annotation_clip=False, color='white', fontsize=12)

        plt.savefig(
            os.path.join(output_folder, file_name),
            dpi=300)
        plt.close(fig)

    return(output_folder+'/'+file_name)

def make_movie_from_commandline(config_dir, results_dir, var_name):
    pwd = os.getcwd()

    plots_dir = results_dir.replace('results', 'plots')
    panels_dir = os.path.join(config_dir, plots_dir, var_name)

    experiment = results_dir.replace('results_', '')

    os.chdir(panels_dir)

    output_name = var_name + '_'+experiment + '.mp4'

    #os.system("ffmpeg -r 5 -i "+var_name+"_"+str(year)+'{:02d}'.format(month)+"%02d.png -vcodec mpeg4 -b 3M -y " + output_name)

    #-pattern_type glob -i
    os.system("ffmpeg -r 5 -pattern_type glob -i '"+var_name+"_*.png' -vcodec mpeg4 -b 3M -y " + output_name)
    os.rename(output_name, os.path.join('..', 'Movies', output_name))

    os.chdir(pwd)

config_dir = '/Volumes/chukchi_sea/Research/Ocean_Modeling/Projects/Chukchi_Sea/'
# config_dir = '/Volumes/ikinngut/Research/Model Results/Chukchi Sea'

project_folder = '/Users/mhwood/Documents/Research/Projects/Chukchi_Sea'

x,y,img = read_MODIS_imagery(project_folder)

XC, YC, Depth = read_model_grid(project_folder)

# reproject the model grid to 32602
points = reproject_points(np.column_stack((XC.flatten(), YC.flatten())), 4326, 32602)
X = points[:, 0].reshape(XC.shape)
Y = points[:, 1].reshape(YC.shape)


for results_dir in ['results_daily_hydrology', 'results_control','results_prescribed_seaice']:
    print('Processing results for the ' + results_dir+' experiment')
    for var_name in ['Theta','Salt','SIheff','SIarea']:
        print('    - Creating movie for ' + var_name)
        experiment = results_dir.replace('results_', '')
        movie_file = var_name + '_'+experiment + '.mp4'
        if movie_file not in os.listdir(os.path.join(config_dir, results_dir.replace('results','plots'), 'Movies')):
            for year in [2023,2024]:#,2024]:
                print('        - Processing year ' + str(year))
                all_file_paths = []
                if year==2024:
                    start_month=1
                else:
                    start_month=2
                for month in range(start_month,13):
                    if month in [1,3,5,7,8,10,12]:
                        days_in_month = 31
                    elif month in [4,6,9,11]:
                        days_in_month = 30
                    elif month == 2:
                        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                            days_in_month = 29
                        else:
                            days_in_month = 28
                    if year==2024 and month==12:
                        days_in_month=29

                    subset, file_name = var_name_to_subset_and_filename(var_name, f'{year}{month:02d}')

                    if file_name in os.listdir(os.path.join(config_dir, results_dir, subset, var_name)):
                        data = read_model_field_from_nc(config_dir, results_dir, subset, var_name, year, month)
                        for day in range(1, days_in_month + 1):
                            date_str = f'{year}{month:02d}{day:02d}'
                            file_path = plot_panel(config_dir, results_dir, var_name, data[day-1,:,:], date_str, x, y, img, Depth)
                            all_file_paths.append(file_path)

            make_movie_from_commandline(config_dir, results_dir, var_name)