

import os
import numpy as np
import netCDF4 as nc4
import matplotlib.pyplot as plt
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

def read_model_grid(project_folder):
    ds = nc4.Dataset(os.path.join(project_folder, 'Data','Model', 'Chukchi_Sea_grid.nc'))
    XC = ds.variables['XC'][:,:]
    YC = ds.variables['YC'][:,:]
    Depth = ds.variables['Depth'][:,:]
    hFacC = ds.variables['HFacC'][:,:]
    ds.close()
    return XC, YC, Depth, hFacC

def compute_runoff_points(hFacC):

    mask = hFacC[0,:,:]
    mask[mask>0]=1

    coastal_mask = np.zeros_like(mask)

    for row in range(1,np.shape(mask)[0]-1):
        for col in range(1,np.shape(mask)[1]-1):
            if mask[row,col] == 1:
                if mask[row-1,col] == 0 or mask[row+1,col] == 0 or mask[row,col-1] == 0 or mask[row,col+1] == 0:
                    coastal_mask[row,col] = 1

    return coastal_mask

def reassign_runoff_points(X, Y, runoff, coastal_mask):

    coastal_rows, coastal_cols = np.where(coastal_mask == 1)
    coastal_x = X[coastal_rows, coastal_cols]
    coastal_y = Y[coastal_rows, coastal_cols]

    coastal_runoff = np.zeros_like(runoff)
    for row in range(1, np.shape(coastal_mask)[0] - 1):
        for col in range(1, np.shape(coastal_mask)[1] - 1):
            if np.any(runoff[:,row, col] > 0):
                x = X[row, col]
                y = Y[row, col]
                distance = np.sqrt((coastal_x - x) ** 2 + (coastal_y - y) ** 2)
                closest_index = np.argmin(distance)
                coastal_runoff[:, coastal_rows[closest_index], coastal_cols[closest_index]] += runoff[:, row, col]

    return coastal_runoff

project_folder = '/Users/mike/Documents/Research/Projects/Chukchi Sea'
input_dir = project_folder + '/Model/input'

grid = np.fromfile(input_dir+'/runoff-2d-Fekete.bin', '>f4').reshape((12, 720, 960))

print('Reading model grid...')
XC, YC, Depth, hFacC = read_model_grid(project_folder)
points = reproject_points(np.column_stack((XC.flatten(), YC.flatten())), 4326, 32602)
X = points[:, 0].reshape(XC.shape)
Y = points[:, 1].reshape(YC.shape)

print('Computing coastal runoff mask...')
coastal_mask = compute_runoff_points(hFacC)

# plt.pcolormesh(coastal_mask, cmap='viridis')
# plt.show()

print('Reassigning runoff points...')
coastal_runoff = reassign_runoff_points(X, Y, grid, coastal_mask)

plt.pcolormesh(coastal_runoff[7,:,:], cmap='viridis', vmin=0, vmax=1e-6)
plt.show()

coastal_runoff.ravel(order='C').astype('>f4').tofile(input_dir+'/runoff-2d-Fekete-coastal.bin')
