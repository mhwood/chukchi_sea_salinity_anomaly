

import numpy as np

f = open('/Users/mike/Documents/Research/Projects/Chukchi Sea/Data/Observations/section_bs_dbo3_full.txt')
lines = f.readlines()
f.close()

all_points = []

for line in lines:
    line = line.strip()
    if line.startswith('6') or line.startswith('7'):
        lineparts = line.split()
        lon_deg = -float(lineparts[2])
        lat_deg = float(lineparts[0])
        lon_min = -float(lineparts[3])
        lat_min = float(lineparts[1])

        lon = lon_deg + lon_min/60
        lat = lat_deg + lat_min/60

        all_points.append([lon, lat])

all_points = np.array(all_points)

all_points += np.array([360, 0])  # Convert to 0-360 longitude

bering_points = all_points[all_points[:,1]<66,:]
for p in range(len(bering_points)):
    print('[',bering_points[p,0],',',bering_points[p,1], '],')
# print(bering_points)

# dbo3_points = all_points[np.logical_and(all_points[:,1]>66, all_points[:,1]<70),:]
# for p in range(len(dbo3_points)):
#     print('[', dbo3_points[p, 0], ',', dbo3_points[p, 1], '],')

# dbo5_points = all_points[all_points[:, 1] > 70, :]
# for p in range(len(dbo5_points)):
#     print('[', dbo5_points[p, 0], ',', dbo5_points[p, 1], '],')