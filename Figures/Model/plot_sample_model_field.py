

import os
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc4
from pyproj import Transformer
import cmocean.cm as cm

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
    ds = nc4.Dataset(os.path.join(project_folder, 'Model','input', 'Chukchi_Sea_grid.nc'))
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


def read_model_field(project_folder, var_name, iter_number):

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


    data = np.fromfile(os.path.join(project_folder,'Model','run','diags',subset,file_name), dtype='>f4')
    data = data.reshape((n_levels, 720, 960))
    data = data[lev,:,:]

    return data

def read_velocity_field(project_folder, file_name):

    depth_level = 5

    data = np.fromfile(os.path.join(project_folder,'Model','run','diags','vel_3D_day_mean',file_name), dtype='>f4')
    data = data.reshape((3*90, 720, 960))
    speed = np.sqrt(data[depth_level,:,:]**2 + data[depth_level+90,:,:]**2)
    U = data[depth_level,:,:]
    V = data[depth_level+90,:,:]

    return U, V, speed

project_folder = '/Users/mhwood/Documents/Research/Projects/Chukchi_Sea/'


x,y,img = read_MODIS_imagery(project_folder)

XC, YC, Depth = read_model_grid(project_folder)

# reproject the model grid to 32602
points = reproject_points(np.column_stack((XC.flatten(), YC.flatten())), 4326, 32602)
X = points[:, 0].reshape(XC.shape)
Y = points[:, 1].reshape(YC.shape)

var_name = 'Speed'
iter_number = 33212160

if var_name not in ['Speed']:
    data = read_model_field(project_folder, var_name, iter_number)
else:
    U, V, speed = read_velocity_field(project_folder, 'vel_3D_day_mean.' + str(iter_number).zfill(10) + '.data')
    data = speed

plot_metadata = {'SIheff':{'units': 'm', 'long_name': 'Sea Ice Thickness','cmap':cm.ice,'vmin': 0, 'vmax': 2},
                 'Theta':{'units': 'C', 'long_name': 'Sea Surface Temperature','cmap':cm.thermal,'vmin': -2, 'vmax': 10},
                 'Salt':{'units': 'psu', 'long_name': 'Salinity','cmap':cm.haline,'vmin': 25, 'vmax': 35},
                 'UVEL':{'units': 'm/s', 'long_name': 'Eastward Velocity','cmap':'viridis','vmin': -0.5, 'vmax': 0.5},
                 'VVEL':{'units': 'm/s', 'long_name': 'Northward Velocity','cmap':'viridis','vmin': -0.5, 'vmax': 0.5},
                 'Speed':{'units': 'm/s', 'long_name': 'Speed','cmap':'viridis','vmin': 0, 'vmax': 0.5}}

# file_name = 'vel_3D_day_mean.0033134400.data'
# U,V, data = read_velocity_field(project_folder, os.path.join(project_folder, 'Model', 'run', file_name))



fig = plt.figure(figsize=(10, 8))
plt.style.use('dark_background')

plot_grid = np.ma.masked_where(Depth == 0, data)

plt.imshow(img, extent=(x.min(), x.max(), y.min(), y.max()), alpha=0.8)
C = plt.imshow(plot_grid, extent=(X.min(), X.max(), Y.min(), Y.max()), cmap=plot_metadata[var_name]['cmap'], origin='lower',
               vmin=plot_metadata[var_name]['vmin'], vmax=plot_metadata[var_name]['vmax'])
plt.colorbar(C, label=plot_metadata[var_name]['units'], orientation='vertical', pad=0.02, aspect=40,
              shrink=0.8, ticks=np.linspace(plot_metadata[var_name]['vmin'], plot_metadata[var_name]['vmax'], 6))

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
plt.title(plot_metadata[var_name]['long_name'] + ' (5m depth)\nSpin-Up Period, Nominal Date 2023/07/29', fontsize=12)#, pad=20)


plt.savefig(os.path.join(project_folder, 'Figures', 'Model Spinup', 'Chukchi_Sea_'+var_name + '_spinup_snapshot.png'),
            dpi=300, bbox_inches='tight')
plt.close(fig)

