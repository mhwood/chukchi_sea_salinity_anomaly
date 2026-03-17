
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import netCDF4 as nc4
import cartopy.crs as ccrs


def read_locations_from_nc(project_folder):

    names = []
    lons = []
    lats = []
    ds = nc4.Dataset(os.path.join(project_folder, '../Data', 'River Discharge', 'Processed',
                                  'alaska_river_flux_2023.nc'))
    for group in ds.groups:
        lons.append(ds[group].longitude)
        lats.append(ds[group].latitude)
        names.append(group)

    return(lons, lats, names)


def plot_locations_on_alaska_map(project_folder, lons, lats, names):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())
    ax.set_extent([-180, -130, 50, 75], crs=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.BORDERS)
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN)

    ax.scatter(lons, lats, color='blue', s=20, transform=ccrs.PlateCarree())

    # label each one with the group name
    for lon, lat, name in zip(lons, lats, names):
        ax.text(lon + 0.5, lat + 0.5, name, fontsize=8, transform=ccrs.PlateCarree())

    plt.title('River Gauge Locations in Alaska')
    plt.show()

project_dir = '/Users/mike/Documents/Research/Projects/Chukchi Sea'

lons, lats, names = read_locations_from_nc(project_dir)

plot_locations_on_alaska_map(project_dir, lons, lats, names)



