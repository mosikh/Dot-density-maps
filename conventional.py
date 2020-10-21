import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import geopandas as gpd
from shapely.geometry import MultiLineString
from pyproj import CRS
import random
from scipy.spatial import distance

## importing the input shp as geodataframe
input_data = gpd.read_file('data\\osm_pois_munich.shp')
input_data = input_data.to_crs("EPSG:31468")

input_data["x"] = input_data.centroid.x
input_data["y"] = input_data.centroid.y

## Running the gaussian KDE function by a n * n grid
xmin = (input_data["x"].min()-5000)
xmax = (input_data["x"].max()+5000)
ymin = (input_data["y"].min()-5000)
ymax = (input_data["y"].max()+5000)


## Making a meshgrid with n * n grids and KDE Gaussian function
## Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
X, Y = np.mgrid[xmin:xmax:23j, ymin:ymax:23j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([input_data["x"], input_data["y"]])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)


## The process of finding the minimum Z value in which there is a point
X_flat=[]
for i in range(len(X)):
    X_flat.append(X[i][0])

Y_flat=list(Y[0])

X_flat.append((X_flat[-1])*2 - X_flat[-2])
Y_flat.append((Y_flat[-1])*2 - Y_flat[-2])

## Making a list from all of the Z_Values
Z_flat = []
for sublist in Z:
    for item in sublist:
        Z_flat.append(item)
        
## Creating a polygon vector from the raster grid
## Source: https://stackoverflow.com/questions/37041377/numpy-meshgrid-to-shapely-polygons
from shapely.ops import polygonize
hlines = [((x1, yi), (x2, yi)) for x1, x2 in list(zip(X_flat[:-1], X_flat[1:])) for yi in Y_flat]
vlines = [((xi, y1), (xi, y2)) for y1, y2 in zip(Y_flat[:-1], Y_flat[1:]) for xi in X_flat]
polys = list(polygonize(MultiLineString(hlines + vlines)))
id = [i for i in range(len(polys))]
grid = gpd.GeoDataFrame({"id":id,"geometry":polys})
grid.crs = CRS.from_epsg(31468).to_wkt()
grid = grid.to_crs("EPSG:31468")


## Adding Z_values of the raster grid to the vector grid as a new column
grid['z_value']=None
for i in range(len(grid)):
    grid.loc[i,'z_value'] = Z_flat[i]
grid['z_value'] = grid['z_value'] + 1

## Rmoving grids which have Z_values equal to zero and are located out of
##the study area
grid = grid[grid.z_value != 1]
grid['z_value'] = grid['z_value'] -1
grid.reset_index(drop=True, inplace=True)

## Calculating zi for each grid which is the first step of Lavin's formula
g_min = grid['z_value'].min()
g_max = grid['z_value'].max()
grid['zi']=None
for i in range(len(grid)):
    grid.loc[i,'zi'] = ((grid.loc[i,'z_value'])- g_min)/(g_max - g_min)

## Calculating pi which is the second step of Lavin's formula
p_max=0.50
p_min=0.01
grid['pi']=None
for i in range(len(grid)):
    grid.loc[i,'pi'] = ((grid.loc[i,'zi'])*(p_max - p_min)) + p_min

## Defining a radius for dots and calculating dot area
radius = 85.00
dot_area = (radius ** 2)*3.14

"""
Calculating the real useful area of grids. We have made a kind of margin
for any grid to avoid overlapping of dots with other dots from neighbor grids
"""
xx,yy =(grid['geometry'].iloc[0]).exterior.coords.xy
grid_area = (xx[0] - xx[1] - radius ) * (yy[2] - yy[1] - radius )

## Calculating dot number in each grid based on the last step of Lavin formula
grid['dot_no']=None
for i in range(len(grid)):
    grid.loc[i,'dot_no'] = round((grid.loc[i,"pi"] * grid_area)/dot_area)

## Making the dots based on the results of formula and avoiding any overlap
g = grid['geometry']
output_list = [[] for x in range(len(g))]

## Dot placement process
## Source: https://github.com/agaidus/census_data_extraction
for j in range(len(g)):
   while len(output_list[j]) < grid.loc[j,"dot_no"]:
      x,y =g[j].exterior.coords.xy
      input_list = [{'x':random.uniform(x[1]+radius/2, x[3]-radius),'y':random.uniform(y[0]+radius/2, y[2]-radius),'radius':radius}]
      for point in input_list:
         circle = plt.Circle((point['x'], point['y']), point['radius'],  fill=True, facecolor='b')
         if len(output_list[j]) == 0:
            output_list[j].append(circle)
         elif all(distance.euclidean(old_circles.center, circle.center) >= (radius*2) for old_circles in output_list[j]):
            output_list[j].append(circle)

## Ploting the dots       
ax = plt.gca(aspect='equal')
ax.cla()

for i in range(len(g)):
    for circle in output_list[i]:    
      ax.add_artist(circle)

dot_value = round(len(input_data)/ grid['dot_no'].sum())


#############################################################################
## The process of making a legend and other details for the map.
legend_elements=[plt.scatter([], [],s=radius/100, marker='o', color='b',facecolor='b',
                        label=str(int(dot_value)))]
plt.legend(
       facecolor='w', 
       edgecolor='k',
       scatterpoints=1,
       ncol=1,
       prop={'family':'Georgia', 'size':8},
       title="Tweets",
       loc='upper center', 
       bbox_to_anchor=(1.15, 1.025), 
       shadow=True)

ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
gfont = {'fontname':'Georgia'}
plt.title('Conventional Dot Density Map of the Oktoberfest Tweets',**gfont)
plt.xlabel('Longitude',**gfont)
plt.ylabel('Latitude',**gfont)
plt.xticks(rotation=45,**gfont)
plt.yticks(rotation=45,**gfont)
plt.tight_layout()
plt.savefig('conventional.png', dpi = 3000)
plt.show()


