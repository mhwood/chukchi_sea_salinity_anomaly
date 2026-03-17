

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

    for month in range(7, 13):
        if month in [1, 3, 5, 7, 8, 10, 12]:
            days_in_month = 31
        elif month in [4, 6, 9, 11]:
            days_in_month = 30
        elif month == 2:
            if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                days_in_month = 29
            else:
                days_in_month = 28
        if year==2023 and month==12:
            days_in_month=28
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

def read_seaice_obs(project_folder, year):
    ds = nc4.Dataset(os.path.join(project_folder,'Data','Observations', 'Sea Ice',
                                  'Chukchi Sea Seaice '+str(year) + '.nc'))
    data = ds.variables['seaice_conc'][:,:, :]
    ds.close()
    return data

def plot_var_panel(ax, x, y, img, Depth, data, var_name, plot_metadata):
    plot_grid = np.ma.masked_where(Depth == 0, data)
    rect = Rectangle((x.min(), y.min()), x.max() - x.min(), y.max() - y.min(), linewidth=1, edgecolor='w',
                     facecolor='white', zorder=0)
    plt.gca().add_patch(rect)
    plt.imshow(img, extent=(x.min(), x.max(), y.min(), y.max()), alpha=0.8, zorder=2)
    C = plt.imshow(plot_grid, extent=(X.min(), X.max(), Y.min(), Y.max()), cmap=plot_metadata[var_name]['cmap'],
                   origin='lower',
                   vmin=plot_metadata[var_name]['vmin'], vmax=plot_metadata[var_name]['vmax'], zorder=3)
    #plt.colorbar(C, label=plot_metadata[var_name]['units'], orientation='vertical', pad=0.02, aspect=40,
    #             shrink=0.8, ticks=np.linspace(plot_metadata[var_name]['vmin'], plot_metadata[var_name]['vmax'], 6))
    plt.contour(X, Y, Depth, levels=[0], colors='w', linewidths=0.5, zorder=4)

    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    # if 'SI' in var_name:
    #     plt.title(plot_metadata[var_name]['long_name'],
    #               fontsize=12)
    # else:
    #     plt.title(plot_metadata[var_name]['long_name'] + ' (5m depth)',
    #               fontsize=12)

    # add labels for Alaska and Russia with semitransparent bounding boxes
    ax.text(x.min() + 0.85 * (x.max() - x.min()), y.min() + 0.65 * (y.max() - y.min()), 'Alaska', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'),
            ha='right', va='center', zorder=5)
    ax.text(x.min() + 0.2 * (x.max() - x.min()), y.min() + 0.6 * (y.max() - y.min()), 'Russia', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'),
            ha='left', va='top', zorder=5)

    # set the extent to remove 3% from edges
    plt.xlim(x.min() + 0.03 * (x.max() - x.min()), x.max() - 0.03 * (x.max() - x.min()))
    plt.ylim(y.min() + 0.03 * (y.max() - y.min()), y.max() - 0.03 * (y.max() - y.min()))

def plot_panel(config_dir, var_name, data_control, data_daily_hydrology, data_prescribed_seaice, data_seaice_obs, date_str, x, y, img, Depth):
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

    file_name = var_name+'_'+ date_str + '.png'
    plots_dir = 'plots/experiment_comparisons'
    if var_name not in os.listdir(os.path.join(config_dir, plots_dir)):
        os.mkdir(os.path.join(config_dir, plots_dir, var_name))
    output_folder = os.path.join(config_dir, plots_dir,var_name)

    if file_name not in os.listdir(output_folder):
        print(f'            - Plotting {var_name} for {date_str} in '+plots_dir)

        day = int(date_str[6:8])

        fig = plt.figure(figsize=(12, 10))
        plt.style.use('dark_background')

        panel_height = 15
        panel_width = 15
        spacing = 1
        timebar_height = 3
        colorbar_width = 1

        gs = GridSpec(nrows=2*panel_height + timebar_height+2*spacing, ncols=2*panel_width + colorbar_width + spacing,
                      figure=fig, left = 0.02, right = 0.95, top = 0.91, bottom = 0.08)

        ###########################################################################################
        # control
        ax = fig.add_subplot(gs[:panel_height, :panel_width])
        plot_var_panel(ax, x, y, img, Depth, data_control, var_name, plot_metadata)
        ax.set_title('Control Experiment', fontsize=12)

        ###########################################################################################
        # daily hydrology
        ax = fig.add_subplot(gs[:panel_height, panel_width+spacing:2*panel_width+spacing])
        plot_var_panel(ax, x, y, img, Depth, data_daily_hydrology, var_name, plot_metadata)
        ax.set_title('Daily Hydrology Experiment', fontsize=12)

        ###########################################################################################
        # prescribed sea ice
        ax = fig.add_subplot(gs[panel_height+2*spacing:2*panel_height+2*spacing, :panel_width])
        plot_var_panel(ax, x, y, img, Depth, data_prescribed_seaice, var_name, plot_metadata)
        ax.set_title('Prescribed Sea Ice Experiment', fontsize=12)

        ###########################################################################################
        # observed
        ax = fig.add_subplot(gs[panel_height + 2*spacing:2 * panel_height + 2*spacing, panel_width+spacing:2*panel_width+spacing])
        plot_var_panel(ax, x, y, img, Depth, data_seaice_obs, var_name, plot_metadata)
        ax.set_title('Observations', fontsize=12)

        ###########################################################################################
        # manual colorbar
        axc = fig.add_subplot(gs[2*spacing:2*panel_height, -colorbar_width])
        x = np.array([0, 1])
        y = np.linspace(plot_metadata[var_name]['vmin'], plot_metadata[var_name]['vmax'], 256)
        Xc, Yc = np.meshgrid(x, y)
        Cc = axc.pcolormesh(Xc, Yc, Yc, cmap=plot_metadata[var_name]['cmap'], vmin=plot_metadata[var_name]['vmin'],
                            vmax=plot_metadata[var_name]['vmax'])
        axc.set_yticks(np.linspace(plot_metadata[var_name]['vmin'], plot_metadata[var_name]['vmax'], 6))
        axc.set_xticks([])
        axc.set_ylabel(plot_metadata[var_name]['units'], fontsize=12)
        # move the colorbar ticks to the right side
        axc.yaxis.set_label_position("right")
        axc.yaxis.tick_right()


        ax3 = fig.add_subplot(gs[-timebar_height+spacing:, 1:-2])
        date = datetime.strptime(date_str, '%Y%m%d')
        min_iter = date_to_iter_number(datetime(int(date_str[:4]), 1, 1), 30)
        max_iter = date_to_iter_number(datetime(int(date_str[:4]) + 1, 1, 1), 30)
        iter_number = date_to_iter_number(date, 30)
        width = (iter_number - min_iter) / (max_iter - min_iter)
        rect = Rectangle((date.year, 0), width, 1, fc='silver', ec='white')
        ax3.add_patch(rect)
        ax3.set_xlim([date.year, date.year + 1])
        ax3.set_ylim([0, 1])
        for i in range(2, 13):
            month_start_iter = date_to_iter_number(datetime(date.year, i, 1), 30)
            x = date.year + (month_start_iter - min_iter) / (max_iter - min_iter)
            plt.plot([x,x], [0, 1], 'w-', linewidth=0.5)
        ax3.set_xticks(np.arange(date.year + 1 / 24, date.year + 1, 1 / 12))
        ax3.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        ax3.set_yticks([])
        ax3.set_xlabel(date_str[:4])

        plt.suptitle('Sea Ice Concentration Comparison')

        plt.savefig(
            os.path.join(output_folder, file_name),
            dpi=300)
        plt.close(fig)

    return(output_folder+'/'+file_name)

def make_movie_from_commandline(config_dir, plots_dir, var_name, year):
    pwd = os.getcwd()

    panels_dir = os.path.join(config_dir, plots_dir, var_name)

    os.chdir(panels_dir)

    output_name = var_name + '_' + str(year)+'_comparison.mp4'

    #os.system("ffmpeg -r 5 -i "+var_name+"_"+str(year)+'{:02d}'.format(month)+"%02d.png -vcodec mpeg4 -b 3M -y " + output_name)

    #-pattern_type glob -i
    os.system("ffmpeg -r 5 -pattern_type glob -i '"+var_name+"_"+str(year)+"*.png' -vcodec mpeg4 -b 3M -y " + output_name)
    os.rename(output_name, os.path.join('..', output_name))

    os.chdir(pwd)

config_dir = '/Volumes/kullorsuaq/Research/Ocean_Modeling/Projects/Chukchi_Sea'
config_dir = '/Users/mike/Documents/Research/Projects/Chukchi Sea/Model'

project_folder = '/Users/mike/Documents/Research/Projects/Chukchi Sea/'

x,y,img = read_MODIS_imagery(project_folder)

XC, YC, Depth = read_model_grid(project_folder)

# reproject the model grid to 32602
points = reproject_points(np.column_stack((XC.flatten(), YC.flatten())), 4326, 32602)
X = points[:, 0].reshape(XC.shape)
Y = points[:, 1].reshape(YC.shape)

var_name = 'SIarea'

experiments = ['daily_hydrology', 'control', 'prescribed_seaice']

for year in [2023]:
    print('        - Processing year ' + str(year))
    all_file_paths = []
    for month in range(2,4):
        if month in [1,3,5,7,8,10,12]:
            days_in_month = 31
        elif month in [4,6,9,11]:
            days_in_month = 30
        elif month == 2:
            if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                days_in_month = 29
            else:
                days_in_month = 28

        subset, file_name = var_name_to_subset_and_filename(var_name, f'{year}{month:02d}')

        data_control = read_model_field_from_nc(config_dir, 'results_control', subset, var_name, year, month)
        data_daily_hydrology = read_model_field_from_nc(config_dir, 'results_daily_hydrology', subset, var_name, year, month)
        data_prescribed_seaice = read_model_field_from_nc(config_dir, 'results_prescribed_seaice', subset, var_name, year, month)
        data_seaice_obs = read_seaice_obs(project_folder, year)

        for day in range(1, days_in_month + 1):
            date_str = f'{year}{month:02d}{day:02d}'

            cumulative_year_day = 0
            for m in range(1, month):
                if m in [1, 3, 5, 7, 8, 10, 12]:
                    cumulative_year_day += 31
                elif m in [4, 6, 9, 11]:
                    cumulative_year_day += 30
                elif m == 2:
                    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                        cumulative_year_day += 29
                    else:
                        cumulative_year_day += 28
            cumulative_year_day += day

            file_path = plot_panel(config_dir, var_name,
                                   data_control[day-1,:,:], data_daily_hydrology[day-1, :, :],
                                   data_prescribed_seaice[day-1,:,:], data_seaice_obs[cumulative_year_day-1,:,:],
                                   date_str, x, y, img, Depth)
            all_file_paths.append(file_path)

make_movie_from_commandline(config_dir, 'plots/experiment_comparisons', var_name, year)