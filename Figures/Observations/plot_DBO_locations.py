
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


def plot_points_on_amp(project_folder, x, y, img, Depth, dbo_point_sets):

    fig = plt.figure(figsize=(10, 6.5))
    plt.style.use('dark_background')

    gs = GridSpec(17, 11, figure=fig, left = 0.02, right = 0.99, top = 0.91, bottom = 0.08)

    ax = fig.add_subplot(gs[:, :])

    plot_grid = np.ma.masked_where(Depth == 0, Depth)

    var_name = 'Depth'
    plot_metadata = {'Depth': {'units': 'm', 'long_name': 'Water Depth', 'cmap': cm.deep, 'vmin': 0, 'vmax': 100}}

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

    colors = {'BS': 'red',
                'DBO3': 'magenta',
                'DBO5': 'yellow'}
    for section in dbo_point_sets:
        points = dbo_point_sets[section]
        plt.plot(points[:, 0], points[:, 1], 'o', markersize=3, label=section,
                 zorder=5, color=colors[section])

    # add legend with white background
    plt.legend(loc='center left', fontsize=12, framealpha=0.5)

    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.title(plot_metadata[var_name]['long_name'],fontsize=12)

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

    plt.savefig(os.path.join(project_folder,'Figures','DBO_Sample_Stations.png'),dpi=300)
    plt.close(fig)



project_folder = '/Users/mike/Documents/Research/Projects/Chukchi Sea/'

x, y, img = read_MODIS_imagery(project_folder)

XC, YC, Depth = read_model_grid(project_folder)

# reproject the model grid to 32602
points = reproject_points(np.column_stack((XC.flatten(), YC.flatten())), 4326, 32602)
X = points[:, 0].reshape(XC.shape)
Y = points[:, 1].reshape(YC.shape)

dbo_sections = ['BS', 'DBO3', 'DBO5']
dbo_point_sets = {}
for section in dbo_sections:
    dbo_points = get_DBO_coordinates(section)
    dbo_points_proj = reproject_points(dbo_points, 4326, 32602)
    dbo_point_sets[section] = dbo_points_proj

plot_points_on_amp(project_folder, x, y, img, Depth, dbo_point_sets)

