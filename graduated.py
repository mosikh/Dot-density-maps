import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import geopandas as gpd
from shapely.geometry import  MultiLineString, Polygon
from pyproj import CRS
import pandas as pd
import math
import jenkspy

## importing the input shp as geodataframe
input_file = gpd.read_file('data\\osm_pois_munich.shp')
input_file = input_file.to_crs("EPSG:31468")

## defining x and y columns based on lat & long of points
input_file["x"] = input_file.centroid.x
input_file["y"] = input_file.centroid.y

## Running the gaussian KDE function by a n * n grid
xmin = (input_file["x"].min()-5000)
xmax = (input_file["x"].max()+5000)
ymin = (input_file["y"].min()-5000)
ymax = (input_file["y"].max()+5000)

## Making a meshgrid with n * n grids and KDE Gaussian function
## Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
X, Y = np.mgrid[xmin:xmax:30j, ymin:ymax:30j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([input_file["x"], input_file["y"]])
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


## Rmoving cells which have Z_values equal to zero and are located out of
##the study area
grid = grid[grid.z_value != 1]
grid['z_value'] = grid['z_value'] -1
grid.reset_index(drop=True, inplace=True)


## Normalization of Z_values to get values between 0 and 1
z_max = grid['z_value'].max()
grid['z_class'] = grid['z_value']/z_max

## Source: https://pbpython.com/natural-breaks.html  
## Using Jenks natural breaks optimization method to classify normalized Z_values
grid['cut_jenksv3'] = pd.cut(
    grid['z_class'],
    bins=jenkspy.jenks_breaks(grid['z_class'], nb_class=6),
    labels=['bucket_1', 'bucket_2', 'bucket_3', 'bucket_4','bucket_5','bucket_6'],
    include_lowest=True)

## Making 6 clusers for our normalized Z_vlaues 
clusters= grid.groupby(['cut_jenksv3'], sort=True)['z_class'].max()

grid["z_class"][(grid["z_class"] <= clusters[0])] = clusters[0]
grid["z_class"][(grid["z_class"] > clusters[0]) & (grid["z_class"] <= clusters[1])] = clusters[1]
grid["z_class"][(grid["z_class"] > clusters[1]) & (grid["z_class"] <= clusters[2])] = clusters[2]
grid["z_class"][(grid["z_class"] > clusters[2]) & (grid["z_class"] <= clusters[3])] = clusters[3]
grid["z_class"][(grid["z_class"] > clusters[3]) & (grid["z_class"] <= clusters[4])] = clusters[4]
grid["z_class"][(grid["z_class"] > clusters[4])] = clusters[5]

## Counting the number of cells within each cluster
count_list = grid.groupby(['cut_jenksv3'], sort=True)['z_class'].count()      

## The process of placing one dot in each cell with diffirent dot sizes. Dot size is based on Z_values
x,y  = grid.loc[1,'geometry'].exterior.coords.xy
r_max = (y[2] - y[1])/2

grid['radius']= None
for i in range(len(grid)):
    grid.loc[i,'radius'] = math.sqrt(grid.loc[i,'z_class'])*(r_max)

## Dot placement process
## Source: https://github.com/agaidus/census_data_extraction
output_list = []
for i in range(len(grid)):
    x=grid.loc[i,'geometry'].centroid.x
    y=grid.loc[i,'geometry'].centroid.y
    input_list = [{'x':x,'y':y,'radius':grid.loc[i,'radius']}]
    for point in input_list:
        circle = plt.Circle((point['x'], point['y']), point['radius'], fill=True, facecolor='b')
        output_list.append(circle)

## Ploting the dots       
ax = plt.gca(aspect='equal')
ax.cla()


for circle in output_list:    
    ax.add_artist(circle)

##########################################################################

"""
Automatically definig levels of contour lines by Jenks natural breaks optimization
"""
levels= grid.groupby(['cut_jenksv3'], sort=True)['z_value'].min()
cs = ax.contour(X, Y, Z, levels=levels, alpha=0)
##########################################################################
"""
The process of counting the number of points of original data within each contour line. Then based on the number
of cells in each contour line, we can measure the dot value for each size of dots.
"""
from shapely.ops import cascaded_union

kde_shp = gpd.GeoDataFrame()
kde_shp['geometry'] = None
list_wkt=[[] for x in range(len(levels))]

for i in range(len(levels)):
    contour = cs.collections[i]
    if range(len(contour.get_paths())) == 1:
        vs=contour.get_paths()[0].vertices
        polygon_vs= Polygon(vs)
        polygon = [polygon_vs]
        kde_shp.loc[len(kde_shp)+1] = polygon
    else:        
        for j in range(len(contour.get_paths())):
            vs=contour.get_paths()[j].vertices
            polygon_vs= Polygon(vs)
            polygon = [polygon_vs]
            list_wkt[i].append(polygon)
        myGeomList = [x[0] for x in list_wkt[i]]
        multi_polygon = cascaded_union(myGeomList)
        multi_pol = [multi_polygon]
        kde_shp.loc[len(kde_shp)+1] = multi_pol

kde_shp.crs = CRS.from_epsg(31468).to_wkt()
kde_shp = kde_shp.to_crs("EPSG:31468")


for i in range(len(kde_shp)-1):
    kde_shp["geometry"].iloc[i] = (kde_shp["geometry"].iloc[i]).difference (kde_shp["geometry"].iloc[i+1])


input_file['region']=-999
for i in range(len(input_file)):
    for j in range(len(kde_shp)):
        if (kde_shp.loc[kde_shp.index[j],'geometry']).contains(input_file.loc[input_file.index[i],'geometry']):
            input_file.loc[input_file.index[i],'region'] = j

input_file["region"] = input_file.region.astype(float)

## calculating the number of points within each contour line
kde_shp['count']= None
for i in range(len(kde_shp)):
    count = input_file.loc[input_file.region == i , 'region'].count()
    kde_shp['count'].iloc[i] = count


dot_values = [i / j for i, j in zip(list(kde_shp['count']), list(count_list))]
dot_values = [math.ceil(num) for num in dot_values]
dot_values.sort()
###########################################################################################################
## The process of making a legend and other details for the map.
sizes = list(grid.radius.unique())
sizes = [number / 10 for number in sizes]
sizes.sort()

labels = np.array(dot_values).astype(str)

size_leg=[]
for i in range (len(clusters)):
    element = plt.scatter([],[], s=sizes[i], marker='o',facecolor='b' , label=labels[i])
    size_leg.append(element)

plt.legend(
       facecolor='w', 
       edgecolor='k',
       scatterpoints=1,
       ncol=1,
       prop={'family':'Georgia', 'size':8},
       title="POIs",
       loc='upper center', 
       bbox_to_anchor=(1.14, 1.025), 
       shadow=True)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
gfont = {'fontname':'Georgia'}
plt.title('Graduated Dot Density Map of POIs of Munich',**gfont)
plt.xlabel('Longitude',**gfont)
plt.ylabel('Latitude',**gfont)
plt.xticks(rotation=45,**gfont)
plt.yticks(rotation=45,**gfont)
plt.tight_layout()
plt.savefig('graduated.png', dpi = 3000)
plt.show()