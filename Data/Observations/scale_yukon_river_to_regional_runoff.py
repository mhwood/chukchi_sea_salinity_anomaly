
import os
import numpy as np
import netCDF4 as nc4
import matplotlib.pyplot as plt
from pyproj import Transformer
import datetime
from scipy.interpolate import griddata
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


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
    rA = ds.variables['rA'][:,:]
    ds.close()
    return XC, YC, Depth, rA

def YMD_to_DecYr(year, month, day, hour=0, minute=0, second=0):
    date = datetime.datetime(year, month, day, hour, minute, second)
    start = datetime.datetime(year, 1, 1)
    end = datetime.datetime(year + 1, 1, 1)
    year_length = (end - start).total_seconds()
    elapsed = (date - start).total_seconds()
    decimal_fraction = elapsed / year_length
    return year + decimal_fraction

def create_time_arrays(year, min_month, max_month):
    time_arrays = []
    total_length = 0
    for month in range(min_month, max_month + 1):
        if month in [1, 3, 5, 7, 8, 10, 12]:
            days_in_month = 31
        elif month in [4, 6, 9, 11]:
            days_in_month = 30
        elif month == 2:
            if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                days_in_month = 29
            else:
                days_in_month = 28

        for day in range(1, days_in_month + 1):
            # for hour in range(24):
            #     for minute in range(0, 60, 15): # every quarter hour
            total_length += 1

    year_array = np.full(total_length, year)
    month_array = np.zeros(total_length, dtype=int)
    day_array = np.zeros(total_length, dtype=int)
    hour_array = np.zeros(total_length, dtype=int)
    minute_array = np.zeros(total_length, dtype=int)
    dec_yr_array = np.zeros(total_length, dtype=float)
    index = 0
    for month in range(min_month, max_month + 1):
        if month in [1, 3, 5, 7, 8, 10, 12]:
            days_in_month = 31
        elif month in [4, 6, 9, 11]:
            days_in_month = 30
        elif month == 2:
            if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                days_in_month = 29
            else:
                days_in_month = 28

        for day in range(1, days_in_month + 1):
            # for hour in range(24):
            #     for minute in range(0, 60, 15):
            month_array[index] = month
            day_array[index] = day
            # hour_array[index] = hour
            # minute_array[index] = minute
            dec_yr_array[index] = YMD_to_DecYr(year, month, day)
            index += 1

    return(year_array, month_array, day_array, dec_yr_array)

def read_river_discharge_from_txt(project_dir, year):

    file_path = os.path.join(project_dir, 'Data','Observations', 'River Discharge', 'Yukon_river_discharge.txt')

    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()

    # Create time arrays for the year
    min_month = 1
    max_month = 12
    years, months, days, dec_yrs = create_time_arrays(year, min_month, max_month)

    station_id_time = dec_yrs
    station_id_discharge = np.zeros_like(years, dtype=float)
    station_id_data_available = np.zeros_like(years, dtype=int)

    counter = 0
    for ll in range(29,len(lines)):
        line = lines[ll]
        line_parts = line.split()
        if len(line_parts)>3:
            # try:
            date_str = line_parts[2]
            discharge_str = line_parts[3]
            discharge = float(discharge_str)
            yr, month, day = map(int, date_str.split('-'))
            # print('Processing:', yr, month, day, 'Discharge:', discharge)
            if year== yr:  # Only process data for the specified year

                # print('Processing:', yr, month, day, 'Discharge:', discharge)

                # Find the index in the time arrays
                row_index = np.where((years == yr) & (months == month) & (days == day))[0]
                #print(year, month, day, hour, minute, '->', row_index, 'Discharge:', discharge)
                if len(row_index) != 0:

                    # Store the data
                    station_id_discharge[row_index[0]] = discharge
                    station_id_data_available[row_index[0]] = 1
                    counter += 1
            # except:
            #     print('Error parsing line:', line.strip())
            #     continue

    # convert from ft3/s to m3/s
    station_id_discharge *= 0.0283168466  # Convert from cubic feet per second to cubic meters per second

    return(station_id_time, station_id_discharge, station_id_data_available)

def read_Blaskey_runoff_data(project_folder):
    ds = nc4.Dataset(os.path.join(project_folder, 'Data', 'Observations','River Discharge',
                                  'Alaska_River_discharge_in_Chukchi_Sea_domain.nc'))
    runoff = ds.variables['runoff'][:,:]
    ds.close()

    runoff = runoff[:,:365]

    return runoff

def read_coastal_climatology_file(project_folder, file_name='runoff-2d-Fekete-coastal.bin'):

    file_path = os.path.join(project_folder, 'Model','input', file_name)

    grid = np.fromfile(file_path, '>f4').reshape((12, 720, 960))

    return grid


project_folder = '/Users/mike/Documents/Research/Projects/Chukchi Sea'

print('Reading model grid...')
XC, YC, Depth, rA = read_model_grid(project_folder)

all_scalars = []

years = np.arange(1992,1996).tolist() + np.arange(2002,2021).tolist()  # Years with Blaskey data

for year in years:

    # print('Processing year:', year)

    # Read the Yukon River discharge data
    # print('Reading Yukon River discharge data...')

    yukon_time, yukon_discharge, yukon_data_available = read_river_discharge_from_txt(project_folder, year)

    blaskey_runoff = read_Blaskey_runoff_data(project_folder)

    scalar = np.sum(blaskey_runoff[year-1992,:365]) / np.sum(yukon_discharge[:365])
    all_scalars.append(scalar)

    # print('    - Scaling factor:', scalar)

    reconstructed_blaskey_discharge = yukon_discharge[:365] * scalar

    RMSE = np.sqrt(np.mean((blaskey_runoff - reconstructed_blaskey_discharge)**2))

    # print('    - RMSE:', RMSE)

    # plt.plot(yukon_time, yukon_discharge, label='Yukon River Discharge', color='blue', linewidth=2.0, alpha=0.7)
    # plt.plot(yukon_time[:365], blaskey_runoff[year-1992,:], label='Blaskey Runoff', color='red', linewidth=2.0, alpha=0.7)
    # plt.plot(yukon_time[:365], reconstructed_blaskey_discharge, label='Reconstructed Blaskey Discharge', color='green', linewidth=2.0, alpha=0.7)
    # plt.legend()
    # plt.show()

print('Average scaling factor over all years:', np.mean(all_scalars), 'p/m', np.std(all_scalars))

print('Reconstructing Blaskey discharge for 2023 and 2024 by scaling Yukon data...')
yukon_time_2023, yukon_discharge_2023, yukon_data_available_2023 = read_river_discharge_from_txt(project_folder, 2023)
reconstructed_blaskey_discharge_2023 = yukon_discharge_2023[:365] * np.mean(all_scalars)
yukon_time_2024, yukon_discharge_2024, yukon_data_available_2024 = read_river_discharge_from_txt(project_folder, 2024)
reconstructed_blaskey_discharge_2024 = yukon_discharge_2024[:365] * np.mean(all_scalars)

# plt.subplot(2,1,1)
# plt.plot(yukon_time_2023[:365]-2023, reconstructed_blaskey_discharge_2023,
#             label='Reconstructed Discharge 2023', color='green', linewidth=2.0, alpha=0.7)
# plt.plot(yukon_time_2024[:365]-2024, reconstructed_blaskey_discharge_2024,
#             label='Reconstructed Discharge 2024', color='orange', linewidth=2.0, alpha=0.7)
# plt.legend()
#
# plt.subplot(2,1,2)
# plt.plot(np.cumsum(reconstructed_blaskey_discharge_2023[152:244]), label='Cumulative Discharge 2023', color='green', linewidth=2.0, alpha=0.7)
# plt.plot(np.cumsum(reconstructed_blaskey_discharge_2024[152:244]), label='Cumulative Discharge 2024', color='orange', linewidth=2.0, alpha=0.7)
# plt.show()

fekete_runoff = read_coastal_climatology_file(project_folder)

fekete_runoff_alaska = np.copy(fekete_runoff)  # Coastal region in Alaska
fekete_runoff_alaska[:,:,:580] = 0  # Set the Russian part to zero

fekete_runoff_russia = np.copy(fekete_runoff)  # Coastal region in Russia
fekete_runoff_russia[:,:,580:] = 0  # Set the Alaska part to zero

fekete_mask_alaska = (fekete_runoff_alaska[6,:,:]>0).astype(int)
coastal_area_alaska = fekete_mask_alaska*rA  # Coastal area in m2
coastal_area_russia = (fekete_runoff_russia[6,:,:]>0).astype(int)*rA  # Coastal area in m2

normalized_fekete_runoff_alaska = fekete_runoff_alaska[6,:,:]/np.sum(fekete_runoff_alaska[6,:,:])
normalized_fekete_runoff_russia = fekete_runoff_russia[6,:,:]/np.sum(fekete_runoff_russia[6,:,:])

runoff_alaska_2023 = np.zeros((365, fekete_runoff_alaska.shape[1], fekete_runoff_alaska.shape[2]))
runoff_alaska_2024 = np.zeros((366, fekete_runoff_alaska.shape[1], fekete_runoff_alaska.shape[2]))
for day in range(365):
    if day < 365:
        runoff_alaska_2023[day,:,:] = normalized_fekete_runoff_alaska * reconstructed_blaskey_discharge_2023[day] / coastal_area_alaska
    runoff_alaska_2024[day,:,:] = normalized_fekete_runoff_alaska * reconstructed_blaskey_discharge_2024[day] / coastal_area_alaska
    if day>150 and day<160:
        print('Day:', day, 'Reconstructed Blaskey Discharge 2023:', reconstructed_blaskey_discharge_2023[day],
              'm3/s, Runoff in Alaska 2023:', runoff_alaska_2023[day,:,580:].sum(),'m/s')
        print('Day:', day, 'Reconstructed Blaskey Discharge 2024:', reconstructed_blaskey_discharge_2024[day],
                'm3/s, Runoff in Alaska 2024:', runoff_alaska_2024[day,:,580:].sum(),'m/s')

fekete_time = np.zeros((14,))
for i in range(12):
    fekete_time[i+1] = YMD_to_DecYr(2002, i + 1, 1)-2002  # Fekete data starts in 2002
fekete_time[0] = fekete_time[1] - 1/12.0  # Set the first time point to be one month before the first month
fekete_time[-1] = fekete_time[-2] + 1/12.0  # Set the last time point to be one month after the last month

print('Interpolating Fekete runoff data to daily resolution...')
fekete_runoff_alaska_daily = np.zeros((365, fekete_runoff.shape[1], fekete_runoff.shape[2]))
dec_yrs = np.linspace(0,1, 365)
for row in range(fekete_runoff.shape[1]):
    for col in range(fekete_runoff.shape[2]):
        if np.any(fekete_runoff[:, row, col] > 0):
            loc_runoff = fekete_runoff[:, row, col]
            # extend loc runoff by one month before and after
            loc_runoff = np.concatenate(([loc_runoff[0]], loc_runoff, [loc_runoff[-1]]))
            fekete_runoff_alaska_daily[:, row, col] = griddata(fekete_time, loc_runoff,
                                                                dec_yrs, method='linear')

# make daily versions for the Alaska and Russia parts
fekete_runoff_alaska_daily = np.copy(fekete_runoff_alaska_daily)
fekete_runoff_russia_daily = np.copy(fekete_runoff_alaska_daily)
fekete_runoff_russia_daily[:,:,580:] = 0  # Set the Alaska part to zero
fekete_runoff_alaska_daily[:,:,:580] = 0  # Set the Russian part to zero
alaska_to_russia_scalar = np.sum(fekete_runoff_russia_daily, axis=(1, 2)) / np.sum(fekete_runoff_alaska_daily, axis=(1, 2))


runoff_2023 = np.copy(runoff_alaska_2023)
runoff_2024 = np.copy(runoff_alaska_2024)
for day in range(365):
    # 2023
    russia_runoff = normalized_fekete_runoff_russia * reconstructed_blaskey_discharge_2023[day] * alaska_to_russia_scalar[day] / coastal_area_russia
    runoff_2023[day,:,:580] = russia_runoff[:,:580]

    # 2024
    russia_runoff = normalized_fekete_runoff_russia * reconstructed_blaskey_discharge_2024[day] * alaska_to_russia_scalar[day] / coastal_area_russia
    runoff_2024[day,:,:580] = russia_runoff[:,:580]

    if day>150 and day<160:
        print('Day:', day, 'Reconstructed Blaskey Discharge 2023:', reconstructed_blaskey_discharge_2023[day],
              'm3/s, Runoff in Russia 2023:', runoff_2023[day,:,:580].sum(),'m/s')
        print('Day:', day, 'Reconstructed Blaskey Discharge 2024:', reconstructed_blaskey_discharge_2024[day],
                'm3/s, Runoff in Russia 2024:', runoff_2024[day,:,:580].sum(),'m/s')

runoff_2023[-1,:,:] = runoff_2023[-2,:,:]  # Fill the last day with the previous day to avoid issues with leap years
runoff_2024[-2,:,:] = runoff_2024[-3,:,:]  # Fill the last day with the previous day to avoid issues with leap years
runoff_2024[-1,:,:] = runoff_2024[-2,:,:]  # Fill the last day with the previous day to avoid issues with leap years


# output the runoff to a binary file
runoff_2023.ravel('C').astype('>f4').tofile(os.path.join(project_folder, 'Model', 'input', 'runoff-2d-Fekete-coastal_2023'))
runoff_2024.ravel('C').astype('>f4').tofile(os.path.join(project_folder, 'Model', 'input', 'runoff-2d-Fekete-coastal_2024'))










