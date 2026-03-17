
import os
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc4
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

def read_runoff_locations(project_folder):
    file_name = os.path.join(project_folder, 'Data', 'Observations','River Discharge','River_Segment_Locations.csv')

    # f = open(file_name, 'r')
    # lines = f.readlines()
    # f.close()
    # lines.pop(0)
    #
    # IDs = []
    # outlets_IDs = []
    # ID_to_location_dict = {}
    # for line in lines:
    #     line = line.strip().split(',')
    #     ID = int(line[0])
    #     lon = float(line[1])
    #     lat = float(line[2])
    #     outlet = int(line[3])
    #     IDs.append(ID)
    #     ID_to_location_dict[ID] = [lon, lat, outlet]
    #     if outlet not in outlets_IDs:
    #         outlets_IDs.append(outlet)

    points = np.loadtxt(file_name, delimiter=',', skiprows=1)
    points[:,1] += 360  # Convert longitude from [-180, 180] to [0, 360]

    # make a boolean array to see if the first and last values are the same
    is_outlet = np.zeros((points.shape[0],), dtype=bool)
    for p in range(points.shape[0]):
        if points[p, 0] == points[p, 3]:
            is_outlet[p] = True

    outlet_points = points[is_outlet, :]  # Extract longitude and latitude of outlet points
    outlet_IDs = outlet_points[:, 0].astype(int)  # Extract IDs of outlet points

    return(outlet_points, outlet_IDs)

def read_model_grid(project_folder):
    ds = nc4.Dataset(os.path.join(project_folder, 'Data','Model', 'Chukchi_Sea_grid.nc'))
    XC = ds.variables['XC'][:,:]
    YC = ds.variables['YC'][:,:]
    Depth = ds.variables['Depth'][:,:]
    rA = ds.variables['rA'][:,:]
    ds.close()
    return XC, YC, Depth, rA

def subsample_outlets_to_coastal_loctions(X, Y, Depth, outlet_points, distance_threshold=10000):

    # get the 0 contour lines of Depth
    fig = plt.figure()
    CS = plt.contour(X, Y, Depth, levels=[0])  # levels specifies the number of contour lines
    plt.close(fig)

    # Access the contour line segments
    # CS.allsegs is a list of lists:
    # - The outer list corresponds to each contour level.
    # - The inner list contains arrays, where each array represents a segment of a contour line.
    # - Each array in the inner list contains (x, y) coordinates of the vertices for that segment.
    all_segments = CS.allsegs[0]

    is_close = np.zeros((outlet_points.shape[0],), dtype=bool)
    for p in range(outlet_points.shape[0]):
        x = outlet_points[p, 1]
        y = outlet_points[p, 2]
        for segment in all_segments:
            distances = np.sqrt((segment[:, 0] - x) ** 2 + (segment[:, 1] - y) ** 2)
            if np.any(distances < distance_threshold):
                is_close[p] = True
                break

    outlet_points = outlet_points[is_close, :]  # Extract only the outlet points that are close to the coastline

    return(outlet_points)

def read_ecco_model_runoff(project_folder):
    grid = np.fromfile(os.path.join(project_folder, 'Model','input', 'runoff-2d-Fekete.bin'), '>f4').reshape((12,720,960))
    # convert m/s to mm/day
    # grid *= 86400 * 1000  # Convert from m/s to mm/day
    return(grid)

def read_daily_runoff_timeseries(project_folder, outlet_ID):

    years = np.arange(1992, 2021, 1).tolist()

    runoff_timeseries = np.zeros((len(years), 366))  # Initialize an array to hold the runoff timeseries

    print('Reading data for '+str(outlet_ID))

    for year in years:
        file_name = 'AK_Rivers_'+str(outlet_ID)+'.h.'+str(year)+'-01-01-43200.nc'
        ds = nc4.Dataset(os.path.join(project_folder, 'Data', 'Observations', 'River Discharge',
                                      'mizuRoute_Output', file_name))
        reachID = ds.variables['reachID'][:]
        runoff = ds.variables['IRFroutedRunoff'][:,:]
        ds.close()

        index = np.where(reachID == outlet_ID)[0]
        runoff_timeseries_year = runoff[:, index]
        runoff_timeseries[years.index(year), :len(runoff_timeseries_year)] = runoff_timeseries_year.flatten()

    return(runoff_timeseries)





    a=1

def write_daily_runoff_to_nc_file(project_folder, daily_runoff_timeseries):

    years = np.arange(1992, 2021, 1)

    file_name = 'Alaska_River_discharge_in_Chukchi_Sea_domain.nc'
    ds = nc4.Dataset(os.path.join(project_folder, 'Data', 'Observations', 'River Discharge',
                                  file_name), 'w', format='NETCDF4')

    ds.createDimension('year', daily_runoff_timeseries.shape[0])
    ds.createDimension('day', daily_runoff_timeseries.shape[1])

    year_var = ds.createVariable('year', 'i4', ('year',))
    year_var[:] = years

    day_var = ds.createVariable('day', 'i4', ('day',))
    day_var[:] = np.arange(1, daily_runoff_timeseries.shape[1] + 1)

    runoff_var = ds.createVariable('runoff', 'f4', ('year', 'day'))
    runoff_var[:] = daily_runoff_timeseries

    ds.close()

# project_folder = '/Users/mhwood/Documents/Research/Projects/Chukchi_Sea'
project_folder = '/Users/mhwood/Documents/Research/Projects/Chukchi_Sea'

XC, YC, Depth, rA = read_model_grid(project_folder)
points = reproject_points(np.column_stack((XC.flatten(), YC.flatten())), 4326, 32602)
X = points[:, 0].reshape(XC.shape)
Y = points[:, 1].reshape(YC.shape)

outlet_points, outlet_IDs = read_runoff_locations(project_folder)
outlet_points = reproject_points(outlet_points, 4326, 32602, x_column=1, y_column=2)

distance_threshold = 50000  # 10 km
outlet_points_close = subsample_outlets_to_coastal_loctions(X, Y, Depth, outlet_points, distance_threshold)

# plt.contour(X, Y, Depth, levels=[0], colors='k', linewidths=0.5)
# plt.scatter(outlet_points[:, 1], outlet_points[:, 2], c='b', s=10, label='Runoff Outlets')
# plt.scatter(outlet_points_close[:, 1], outlet_points_close[:, 2], c='g', s=15, label='Runoff Outlets')
# plt.show()

model_runoff = read_ecco_model_runoff(project_folder)
for i in range(model_runoff.shape[0]):
    model_runoff[i, :, :] *= rA  # Convert from m/s to m3/s

outlet_IDs = outlet_points_close[:, 0].astype(int)
for i in range(len(outlet_IDs)):
    outlet_IDs[i] = int(outlet_IDs[i])
    if i==0:
        daily_runoff_timeseries = read_daily_runoff_timeseries(project_folder, outlet_IDs[i])
    else:
        daily_runoff_timeseries += read_daily_runoff_timeseries(project_folder, outlet_IDs[i])
daily_runoff_climatology = np.mean(daily_runoff_timeseries[:,:365], axis=0)

# subset to alaska only
model_runoff_alaska = model_runoff[:, :, 580:]
model_runoff_russia = model_runoff[:, :, :580]

model_runoff_alaska_timeseries = np.sum(model_runoff_alaska, axis=(1,2))
model_runoff_russia_timeseries = np.sum(model_runoff_russia, axis=(1,2))
model_runoff_timeseries = np.sum(model_runoff, axis=(1,2))

plt.subplot(2,1,1)
plt.plot(model_runoff_alaska_timeseries, label='Model Runoff Alaska')
plt.plot(model_runoff_russia_timeseries, label='Model Runoff Russia')
plt.plot(model_runoff_timeseries, label='Total Model Runoff')
plt.legend()

plt.subplot(2,1,2)
plt.plot(daily_runoff_climatology, label='Runoff Climatology from Observations')
plt.show()

write_daily_runoff_to_nc_file(project_folder, daily_runoff_timeseries)



