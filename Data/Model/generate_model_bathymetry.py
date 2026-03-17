
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py


def read_bathymetry_from_mat_file(file_path):

    f = h5py.File(file_path, 'r')
    depth = f.get('depth')
    ice_shelf = f.get('ice_shelf')
    depth = np.array(depth)

    return(depth, ice_shelf)


project_dir = '/Users/mike/Documents/Research/Projects/Greenland Model Analysis/' \
              'Greenland/Data'

bathy_mat_file = os.path.join(project_dir, 'pan_greenland_bathy_ice_shelf.mat')

depth, ice_shelf = read_bathymetry_from_mat_file(bathy_mat_file)

depth.ravel('C').astype('>f4').tofile(project_dir+'/Pan_Greenland_bathymetry_unfilled.bin')

# plt.subplot(1,2,1)
C = plt.pcolormesh(depth, cmap='Blues', vmin=0, vmax=1000)
plt.colorbar(C)
plt.title('Depth (m)')
# plt.subplot(1,2,2)
# C = plt.pcolormesh(ice_shelf)
# plt.colorbar(C)
# plt.title('Ice Shelf')
plt.show()