
import os
import numpy as np
import netCDF4 as nc4
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import cmocean.cm as cm
from pyproj import Transformer

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

def read_model_grid(config_dir):
    grid_file = os.path.join(config_dir,'Data', 'Model', 'Chukchi_Sea_grid.nc')
    ds = nc4.Dataset(grid_file)

    lon = ds.variables['XC'][:,:]
    lat = ds.variables['YC'][:,:]
    depth = ds.variables['Depth'][:,:]
    hFacC = ds.variables['HFacC'][:,:,:]
    drF = ds.variables['drF'][:]

    ds.close()

    return lon, lat, depth, hFacC, drF

def read_SSS_anomaly_from_nc(project_dir, month):
    ds = nc4.Dataset(os.path.join(project_dir,'Data', 'Observations', 'SMAP_SSS_Anomaly_2025_'+month+'.nc'))
    X = ds.variables['Longitude'][:,:]
    Y = ds.variables['Latitude'][:,:]
    SSS_anomaly = ds.variables['SSS_anomaly'][:,:]
    ds.close()
    return X, Y, SSS_anomaly

def plot_panel(project_folder, x, y, img, X, Y, Depth, X_SSS, Y_SSS, SSS_anomaly, month):

    file_name = 'Chukchi_Sea_SSS_Anomaly_'+month+'.png'
    output_folder = os.path.join(project_folder, 'Figures', 'Observations')

    fig = plt.figure(figsize=(10, 7))
    plt.style.use('dark_background')

    cmap = cm.balance
    vmin = -2
    vmax = 2
    units = 'psu'

    gs = GridSpec(17, 11, figure=fig, left = 0.02, right = 0.99, top = 0.91, bottom = 0.08)

    ax = fig.add_subplot(gs[:, :])

    plot_grid = np.ma.masked_where(SSS_anomaly == 0, SSS_anomaly)

    # add a white Rectangle
    rect = Rectangle((x.min(), y.min()), x.max() - x.min(), y.max() - y.min(), linewidth=1, edgecolor='w',
                     facecolor='white', zorder=0)
    plt.gca().add_patch(rect)

    plt.imshow(img, extent=(x.min(), x.max(), y.min(), y.max()), alpha=0.8, zorder=2)
    C = plt.imshow(plot_grid, extent=(X_SSS.min(), X_SSS.max(), Y_SSS.min(), Y_SSS.max()), cmap=cmap,
                   origin='lower',
                   vmin=vmin, vmax=vmax, zorder=3)
    plt.colorbar(C, label=units, orientation='vertical', pad=0.02, aspect=40,
                 shrink=0.8, ticks=np.linspace(vmin, vmax, 6))

    # add a white contour on the coastlines
    plt.contour(X, Y, Depth, levels=[0], colors='w', linewidths=0.5, zorder=4)


    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.title('SMAP SSS Anomaly - '+month +' 2024 vs 2015-2023 Average',
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

    plt.savefig(
        os.path.join(output_folder, file_name),
        dpi=300)
    plt.close(fig)

    return(output_folder+'/'+file_name)

home_dir = os.path.expanduser('~')

project_dir = home_dir+'/Documents/Research/Projects/Chukchi_Sea'

month = 'September'

XC, YC, Depth, hFacC, drF = read_model_grid(project_dir)

points = reproject_points(np.column_stack((XC.flatten(), YC.flatten())), 4326, 32602)
X = points[:, 0].reshape(XC.shape)
Y = points[:, 1].reshape(YC.shape)

x,y,img = read_MODIS_imagery(project_dir)

X_SSS, Y_SSS, SSS_anomaly = read_SSS_anomaly_from_nc(project_dir, month)

plot_panel(project_dir, x, y, img, X, Y, Depth, X_SSS, Y_SSS, SSS_anomaly, month)



