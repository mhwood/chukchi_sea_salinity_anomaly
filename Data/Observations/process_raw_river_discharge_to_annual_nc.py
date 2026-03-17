
import os
import time

import numpy as np
import netCDF4 as nc4
import matplotlib.pyplot as plt
import datetime
import requests

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
            for hour in range(24):
                for minute in range(0, 60, 15): # every quarter hour
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
            for hour in range(24):
                for minute in range(0, 60, 15):
                    month_array[index] = month
                    day_array[index] = day
                    hour_array[index] = hour
                    minute_array[index] = minute
                    dec_yr_array[index] = YMD_to_DecYr(year, month, day, hour, minute)
                    index += 1

    return(year_array, month_array, day_array, hour_array, minute_array, dec_yr_array)

def read_river_discharge_from_txt(project_dir, file_name, year, min_month, max_month):

    file_path = os.path.join(project_dir, '', 'River Discharge', 'Raw', file_name)

    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()

    # Extract station information
    station_start = 'Data for the following 263 site(s)'
    n_stations = 263
    for ll in range(len(lines)):
        if station_start in lines[ll]:
            station_lines = lines[ll+1:ll+1+n_stations]
            break

    stations = []
    for line in station_lines:
        parts = line.split()
        station_id = parts[2]
        station_name = ' '.join(parts[3:-1])
        stations.append((station_id, station_name))

    # Make time arrays
    years, months, days, hours, minutes, dec_yrs = create_time_arrays(year, min_month, max_month)

    # Extract data
    station_data = {}
    for s, station in enumerate(stations):
        station_id = station[0]
        print('Gathering data for station:', station, '(',s+1, 'of', len(stations), ')')
        station_id_time = dec_yrs
        station_id_discharge = np.zeros_like(years, dtype=float)
        station_id_data_available = np.zeros_like(years, dtype=int)

        counter = 0
        for line in lines:
            line_parts = line.split()
            if len(line_parts)>5:
                if line_parts[1] == station_id:
                    try:
                        date_str = line_parts[2]
                        hour_str = line_parts[3]
                        timezone = line_parts[4]
                        discharge_str = line_parts[5]
                        year, month, day = map(int, date_str.split('-'))
                        hour, minute = map(int, hour_str.split(':'))
                        discharge = float(discharge_str)

                        # Find the index in the time arrays
                        row_index = np.where((years == year) & (months == month) & (days == day) & (hours == hour) & (minutes == minute))[0]
                        #print(year, month, day, hour, minute, '->', row_index, 'Discharge:', discharge)
                        if len(row_index) != 0:

                            # Store the data
                            station_id_discharge[row_index[0]] = discharge
                            station_id_data_available[row_index[0]] = 1
                            counter += 1
                    except:
                        print('Error parsing line:', line.strip())
                        continue

        # print('   - Collected', np.sum(station_id_data_available), 'data points for station', station_id)
        station_data[station_id] = {
            'time': station_id_time,
            'discharge': station_id_discharge,
            'data_available': station_id_data_available,
            'station_name': station[1],
        }

    return(station_data)

def read_location_from_usgs_page(station_id):
    try:
        url = 'https://waterdata.usgs.gov/monitoring-location/USGS-'+station_id
        response = requests.get(url)
        lines = response.text.split('\n')
        longitude = 0
        latitude = 0
        for line in lines:
            if 'LONGITUDE:' in line:
                longitude = float(line.split(':')[1].strip()[1:-1])
            if 'LATITUDE:' in line:
                latitude = float(line.split(':')[1].strip()[1:-2])

        #sleep for a short time to avoid overwhelming the server
        time.sleep(1)
    except:
        print('Error fetching location for station', station_id)
        longitude = 0
        latitude = 0
    return(longitude, latitude)

def write_annual_data_to_nc(project_dir, annual_data, year):
    output_file = os.path.join(project_dir, '', 'River Discharge', 'Processed', 'alaska_river_flux_' + str(year) + '.nc')
    with nc4.Dataset(output_file, 'w', format='NETCDF4') as ds:

        # Create an overall dimension for time the same length as the first time array
        ds.createDimension('time', len(annual_data[next(iter(annual_data))]['time']))

        # Create a global time variable
        time_var = ds.createVariable('time', 'f8', ('time',))
        time_var.units = 'decimal year'
        time_var[:] = annual_data[next(iter(annual_data))]['time']

        # Create dimensions
        for station_id in annual_data.keys():
            group = ds.createGroup(station_id)

            # Create variables
            discharge_var = group.createVariable('discharge', 'f4', ('time',))
            data_available_var = group.createVariable('flag', 'i4', ('time',))

            # Set group attributes
            group.station_name = annual_data[station_id]['station_name']
            group.longitude = annual_data[station_id].get('longitude', 0)
            group.latitude = annual_data[station_id].get('latitude', 0)

            # Assign data to variables
            discharge_var[:] = annual_data[station_id]['discharge']
            data_available_var[:] = annual_data[station_id]['data_available']

            data_available_var.note = '1 = data available, 0 = no data'

        # Add attributes
        ds.description = 'Annual river discharge data for Alaska rivers'
        ds.source = 'USGS Alaska River Discharge Data'
        ds.url = 'https://waterdata.usgs.gov/ak/nwis/uv'



project_dir = '/Users/mike/Documents/Research/Projects/Chukchi Sea'


file_names_2023 = ['alaska_river_flux_202301_202306.txt',
                   'alaska_river_flux_202307_202312.txt']

file_names_2024 = ['alaska_river_flux_202401_202406.txt',
                   'alaska_river_flux_202407_202412.txt']

for year in [2024]:
    if year == 2023:
        file_names = file_names_2023
    if year == 2024:
        file_names = file_names_2024

    station_data_0 = \
        read_river_discharge_from_txt(project_dir, file_names[0], year=year, min_month=1, max_month=6)

    station_data_1 = \
        read_river_discharge_from_txt(project_dir, file_names[1], year=year, min_month=7, max_month=12)

    # concatenate the data
    station_data = {}
    for station_id in station_data_0.keys():
        if station_id in station_data_1:

            # Get the longitude and latitude from the USGS page
            longitude, latitude = read_location_from_usgs_page(station_id)

            full_data_available = np.concatenate((station_data_0[station_id]['data_available'],
                                                    station_data_1[station_id]['data_available']))

            if np.sum(full_data_available) == 0:
                print('Station', station_id, 'has no data for '+str(year))
            elif longitude == 0 or latitude == 0:
                print('Station', station_id, 'has no coordinates on the USGS page')
            else:
                station_data[station_id] = {
                    'time': np.concatenate((station_data_0[station_id]['time'],
                                            station_data_1[station_id]['time'])),
                    'discharge': np.concatenate((station_data_0[station_id]['discharge'],
                                                 station_data_1[station_id]['discharge'])),
                    'data_available': np.concatenate((station_data_0[station_id]['data_available'],
                                                      station_data_1[station_id]['data_available']))
                }

                station_data[station_id]['station_name'] = station_data_0[station_id]['station_name']
                station_data[station_id]['longitude'] = longitude
                station_data[station_id]['latitude'] = latitude

                print('Station', station_id, 'has data for '+str(year), np.sum(station_data[station_id]['data_available']), 'data points')
                print('    - Longitude:', longitude, 'Latitude:', latitude)

    write_annual_data_to_nc(project_dir, station_data, year)

    # plot the first station discharge timeseries
    station_id = list(station_data.keys())[0]
    plt.figure(figsize=(10, 6))
    plt.plot(station_data[station_id]['time'], station_data[station_id]['discharge'], linestyle='-')
    plt.title('River Discharge Time Series for Station ' + station_id)
    plt.show()




