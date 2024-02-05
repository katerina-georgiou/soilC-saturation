
# coding: utf-8

# In[ ]:

###########################################################################
### Global stocks and capacity of mineral-associated soil organic carbon
### Published in Nature Communications, 2022.
### 
### Python script for global MOC & MOCmax analysis
###
### Corresponding data available at https://doi.org/10.5281/zenodo.5987415
### Gridded products available at https://doi.org/10.5281/zenodo.6539765
### All covariates are freely available in the references detailed in the
### manuscript, and are also available from the authors upon request.
###
### Contact: Katerina Georgiou (georgiou1@llnl.gov) with any questions.
###########################################################################


# In[ ]:

###########################################################################
### SET-UP AND DATASETS
###########################################################################


# In[1]:

###########################################################################
### importing packages
import numpy as np
import math
import os
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import csv
import pandas as pd
import numpy.ma as ma
from netCDF4 import Dataset
from mpl_toolkits import basemap
import matplotlib as ml
from mpl_toolkits.basemap import Basemap
import warnings
warnings.filterwarnings('ignore')
import pyproj    
import shapely
import shapely.ops as ops
from shapely.geometry.polygon import Polygon
from functools import partial
from pyproj import Proj
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"
import h5py # for HDF4 use: from pyhdf.SD import SD, SDC
from pylab import *
import matplotlib.colors as colors


# In[2]:

from sklearn.model_selection import train_test_split
from scipy import spatial
from sklearn.metrics import mean_squared_error
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import tree
from sklearn.externals.six import StringIO  
import pydot 
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
#
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.preprocessing import Imputer
#
from sklearn.metrics import r2_score
from scipy.interpolate import make_interp_spline, BSpline, spline
#
import shapely
import shapely.ops as ops
from shapely.geometry.polygon import Polygon
from functools import partial


# In[3]:

def pred_ints(model, X, percentile):
    err_down = []
    err_up = []
    for x in range(len(X)):
        preds = []
        for pred in model.estimators_:
            preds.append(pred.predict(X)[x])
        err_down.append(np.percentile(preds, (100 - percentile) / 2. ))
        err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
    return err_down, err_up


# In[4]:

# map of grid cell area
dlatout = 0.5 # size of lat grid
dlonout = 0.5 # size of lon grid

latsize = int(180/dlatout) # as integer
lonsize = int(360/dlonout) # as integer

area = np.zeros((latsize,lonsize,))
outlats = np.arange(90-dlatout/2, -90, -dlatout)
outlons = np.arange(-180+dlonout/2, 180, dlonout)

for lato in np.arange(0,(latsize-1),1):
    ymax = outlats[lato] + dlatout/2
    ymin = outlats[lato] - dlatout/2
    xmax = outlons[0] + dlonout/2
    xmin = outlons[0] - dlonout/2
    
    geom = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)])
    geom_area = ops.transform(partial(pyproj.transform, pyproj.Proj(init='EPSG:4326'),
                pyproj.Proj(proj='aea', lat1=geom.bounds[1], lat2=geom.bounds[3])), geom)
    
    area[lato,:] = geom_area.area # area in m^2
    
outlats_ud = np.flipud(outlats)
area_ud = np.flipud(area)


# In[5]:

###########################################################################
### setting directory
os.chdir("/Users/")
print(os.getcwd())


# In[6]:

###########################################################################
### load temperature data from CRU
cru_datafilename = 'cru_ts_3.1_climo_1961-1990.nc'
#
print( ' opening file '+cru_datafilename)
cru_datafile = Dataset(cru_datafilename)
cru_data = cru_datafile.variables['tmp'][:].mean(axis=0) 
cru_lats = cru_datafile.variables['lat'][:]
cru_lons = cru_datafile.variables['lon'][:]
cru_datafile.close()
#
lats_common = cru_lats.copy()
lons_common = cru_lons.copy()
#
del cru_lats, cru_lons


# In[7]:

###########################################################################
# common mesh grid (0.5 x 0.5)
nlons, nlats = np.meshgrid(lons_common, lats_common)


# In[8]:

###########################################################################
###  load the precip data from GPCC
gpcc_filename = 'precip.mon.total.v6.nc'
#
print( ' opening file '+gpcc_filename)
gpcc_file = Dataset(gpcc_filename)
gpcc_lats = gpcc_file.variables['lat'][:]
gpcc_lons = gpcc_file.variables['lon'][:]
gpcc_time = gpcc_file.variables['time'][:] # days since 1800
start_date_index = 960                          
gpcc_data = gpcc_file.variables['precip'][start_date_index:,::-1,:].mean(axis=0)*12 # (mm/yr)
gpcc_file.close()
# 
gpcc_data_temp = gpcc_data.copy()
gpcc_data[:,0:360] = gpcc_data_temp[:,360:] # match to (lons_common, lats_common) 
gpcc_data[:,360:] = gpcc_data_temp[:,0:360]
#
del gpcc_lats, gpcc_lons, gpcc_time, gpcc_data_temp


# In[9]:

###########################################################################
### load the clay + silt and bulk density data from HWSD
clayfilename_t = "T_CLAY.nc4"
print(' opening file: '+clayfilename_t)
clayfile_t = Dataset(clayfilename_t, format='NETCDF4')
claylats = clayfile_t.variables['lat'][:]
claylons = clayfile_t.variables['lon'][:]
claymap_in_t = clayfile_t.variables['T_CLAY'][:] 
clayfile_t.close()
#
claymap_halfdegree_t = basemap.interp(claymap_in_t, claylons, claylats, nlons, nlats, order=1) 
#
siltfilename_t = "T_SILT.nc4"
print(' opening file: '+siltfilename_t)
siltfile_t = Dataset(siltfilename_t, format='NETCDF4')
siltlats = siltfile_t.variables['lat'][:]
siltlons = siltfile_t.variables['lon'][:]
siltmap_in_t = siltfile_t.variables['T_SILT'][:] 
siltfile_t.close()
#
siltmap_halfdegree_t = basemap.interp(siltmap_in_t, siltlons, siltlats, nlons, nlats, order=1) 
#
clayfilename_s = "S_CLAY.nc4"
print(' opening file: '+clayfilename_s)
clayfile_s = Dataset(clayfilename_s, format='NETCDF4')
claymap_in_s = clayfile_s.variables['S_CLAY'][:] 
clayfile_s.close()
#
claymap_halfdegree_s = basemap.interp(claymap_in_s, claylons, claylats, nlons, nlats, order=1) 
#
siltfilename_s = "S_SILT.nc4"
print(' opening file: '+siltfilename_s)
siltfile_s = Dataset(siltfilename_s, format='NETCDF4')
siltmap_in_s = siltfile_s.variables['S_SILT'][:] 
siltfile_s.close()
#
siltmap_halfdegree_s = basemap.interp(siltmap_in_s, siltlons, siltlats, nlons, nlats, order=1) 
#
bdfilename = "T_BULK_DEN.nc4"
print(' opening file: '+bdfilename)
bdfile = Dataset(bdfilename, format='NETCDF4')
bdlats = bdfile.variables['lat'][:]
bdlons = bdfile.variables['lon'][:]
bdmap_in = bdfile.variables['T_BULK_DEN'][:] 
bdfile.close()
#
bd_t = basemap.interp(bdmap_in, bdlons, bdlats, nlons, nlats, order=1) 
#
bdsfilename = "S_BULK_DEN.nc4"
print(' opening file: '+bdsfilename)
bdsfile = Dataset(bdsfilename, format='NETCDF4')
bdsmap_in = bdsfile.variables['S_BULK_DEN'][:] 
bdsfile.close()
#
bd_s = basemap.interp(bdsmap_in, bdlons, bdlats, nlons, nlats, order=1) 
#
texture_t = claymap_halfdegree_t + siltmap_halfdegree_t
texture_s = claymap_halfdegree_s + siltmap_halfdegree_s
#
del claymap_halfdegree_t, siltmap_halfdegree_t, claymap_halfdegree_s, siltmap_halfdegree_s, claymap_in_t, siltmap_in_t, claymap_in_s, siltmap_in_s
del claylats, claylons, siltlats, siltlons, bdlats, bdlons, bdmap_in, bdsmap_in


# In[10]:

###########################################################################
### load HWSD soc data
soctfilename = "T_OC.nc4"
print(' opening file: '+soctfilename)
soctfile = Dataset(soctfilename, format='NETCDF4')
soctlats = soctfile.variables['lat'][:]
soctlons = soctfile.variables['lon'][:]
soctmap_in = soctfile.variables['T_OC'][:] 
soctfile.close()
#
soctmap_halfdegree = basemap.interp(soctmap_in, soctlons, soctlats, nlons, nlats, order=1) 
#
socsfilename = "S_OC.nc4"
print(' opening file: '+socsfilename)
socsfile = Dataset(socsfilename, format='NETCDF4')
socslats = socsfile.variables['lat'][:]
socslons = socsfile.variables['lon'][:]
socsmap_in = socsfile.variables['S_OC'][:]
socsfile.close()
#
socsmap_halfdegree = basemap.interp(socsmap_in, socslons, socslats, nlons, nlats, order=1) 
#
del soctlats, soctlons, soctmap_in, socslats, socslons, socsmap_in
#
### calculating soil concentrations
soc_t_hwsd = soctmap_halfdegree *10 # into gC/kg soil
soc_s_hwsd = socsmap_halfdegree *10
#
del soctmap_halfdegree, socsmap_halfdegree


# In[11]:

###########################################################################
### load SoilGrids soc data
soct_datafilename = 'OCSTHA_M_30cm_10km_ll.nc'
#
print( ' opening file '+soct_datafilename) # total to 30 cm
soct_datafile = Dataset(soct_datafilename)
soct_data = soct_datafile.variables['Band1'][:] 
soct_lats = soct_datafile.variables['lat'][:]
soct_lons = soct_datafile.variables['lon'][:]
soct_datafile.close()
#
socs_datafilename = 'OCSTHA_M_100cm_10km_ll.nc' # total to 1 m
#
print( ' opening file '+socs_datafilename)
socs_datafile = Dataset(socs_datafilename)
socs_data = socs_datafile.variables['Band1'][:] 
socs_lats = socs_datafile.variables['lat'][:]
socs_lons = socs_datafile.variables['lon'][:]
socs_datafile.close()
#
soct_data_kgm2 = (soct_data)/10 # from tC/ha
socs_data_kgm2 = (socs_data - soct_data)/10 # from tC/ha
#
# regridding
soct_soilgrids_kgm2 = basemap.interp(soct_data_kgm2, soct_lons, soct_lats, nlons, nlats, order=1) 
socs_soilgrids_kgm2 = basemap.interp(socs_data_kgm2, socs_lons, socs_lats, nlons, nlats, order=1) 
soct_soilgrids_kgm2 = np.ma.masked_less_equal(soct_soilgrids_kgm2, -1)
socs_soilgrids_kgm2 = np.ma.masked_less_equal(socs_soilgrids_kgm2, -1)
#
del soct_data_kgm2, socs_data_kgm2, soct_lons, soct_lats, socs_lons, socs_lats, soct_data, socs_data
#
### calculating soil concentrations
depth=30
BD = bd_t
#
soc_t_soilgrids_conc = soct_soilgrids_kgm2/BD/(depth/10) # converting with BD and depth
soc_t_soilgrids = soc_t_soilgrids_conc*10 # to gC/kg
#
depth=70
BD = bd_s
#
soc_s_soilgrids_conc = socs_soilgrids_kgm2/BD/(depth/10) # converting with BD and depth 
soc_s_soilgrids = soc_s_soilgrids_conc*10 # to gC/kg
#
del soc_t_soilgrids_conc, soc_s_soilgrids_conc, soct_soilgrids_kgm2, socs_soilgrids_kgm2


# In[12]:

###########################################################################
### load clay type data from Ito & Wagai 2017
typefilename = "clay_fraction_hd_v1r1.nc4"
print(' opening file: '+typefilename)
typefile = Dataset(typefilename, format='NETCDF4')
#
kaolinite_t = np.flipud(typefile.variables['kaolinite (topsoil)'][0][:][:]) # flipping so lats match other datasets
smectite_t = np.flipud(typefile.variables['smectite (topsoil)'][0][:][:]) 
vermiculite_t = np.flipud(typefile.variables['vermiculite (topsoil)'][0][:][:])
illite_mica_t = np.flipud(typefile.variables['illite,mica (topsoil)'][0][:][:])
chlorite_t = np.flipud(typefile.variables['chlorite (topsoil)'][0][:][:])
gibbsite_t = np.flipud(typefile.variables['gibbsite (topsoil)'][0][:][:])
#
kaolinite_s = np.flipud(typefile.variables['kaolinite (subsoil)'][0][:][:])
smectite_s = np.flipud(typefile.variables['smectite (subsoil)'][0][:][:])
vermiculite_s = np.flipud(typefile.variables['vermiculite (subsoil)'][0][:][:])
illite_mica_s = np.flipud(typefile.variables['illite,mica (subsoil)'][0][:][:])
chlorite_s = np.flipud(typefile.variables['chlorite (subsoil)'][0][:][:])
gibbsite_s = np.flipud(typefile.variables['gibbsite (subsoil)'][0][:][:])
#
typefile.close()
#
### calculating high and low activity clay (HAC and LAC)
LAC_t = kaolinite_t + gibbsite_t
HAC_t = smectite_t + illite_mica_t + vermiculite_t + chlorite_t
#
LAC_s = kaolinite_s + gibbsite_s
HAC_s = smectite_s + illite_mica_s + vermiculite_s + chlorite_s
#
# calculating dominant clay activity (HAC = 1 when true) 
activity_t = HAC_t > LAC_t 
activity_s = HAC_s > LAC_s
#
activity_t = activity_t*1
activity_s = activity_s*1
#
del kaolinite_t, smectite_t, vermiculite_t, illite_mica_t, chlorite_t, gibbsite_t
del kaolinite_s, smectite_s, vermiculite_s, illite_mica_s, chlorite_s, gibbsite_s


# In[13]:

###########################################################################
#### load NCSCD organic soil maps
histel_map = 'NCSCD_Circumarctic_histel_pct_05deg.nc'
histosol_map = 'NCSCD_Circumarctic_histosol_pct_05deg.nc'
#
print(' opening file: '+histel_map)
histel_file = Dataset(histel_map)
histel_pct = histel_file.variables['NCSCD_Circumarctic_histel_pct_05deg.tif'][:]
histel_lats = histel_file.variables['lat'][:]
histel_lons = histel_file.variables['lon'][:]
histel_file.close()
#
print(' opening file: '+histosol_map)
histosol_file = Dataset(histosol_map)
histosol_pct = histosol_file.variables['NCSCD_Circumarctic_histosol_pct_05deg.tif'][:]
histosol_lats = histosol_file.variables['lat'][:]
histosol_lons = histosol_file.variables['lon'][:]
histosol_file.close()
#
max_hist_frac = 50.
#
organics = histel_pct.astype('float') + histosol_pct.astype('float')
#
organics_lat_offset = int((histosol_lats.min() - lats_common.min())*2) 
IM_common = len(lons_common)
JM_common = len(lats_common)
organics_commonmap = np.zeros([JM_common, IM_common], dtype=np.bool)
organics_commonmap[organics_lat_offset:organics_lat_offset+len(histosol_lats),:] = organics[::-1,:] > max_hist_frac
organics_commonmap = np.ma.masked_array(organics_commonmap)


# In[14]:

###########################################################################
### loading land-use type data from Harden et al. 2017
rootgrp = Dataset('crop.nc', 'r', format='NETCDF4')
lon_lu = rootgrp.variables['lon'][:]
lat_lu = np.flipud(rootgrp.variables['lat'][:])
frac_crop = np.flipud(rootgrp.variables['frac'][0][:][:])
rootgrp.close()
#
rootgrp = Dataset('unused.nc', 'r', format='NETCDF4')
frac_unused = np.flipud(rootgrp.variables['frac'][0][:][:])
rootgrp.close()
#
rootgrp = Dataset('forestry.nc', 'r', format='NETCDF4')
frac_forestry = np.flipud(rootgrp.variables['frac'][0][:][:])
rootgrp.close()
#
rootgrp = Dataset('grazing.nc', 'r', format='NETCDF4')
frac_grazing = np.flipud(rootgrp.variables['frac'][0][:][:])
rootgrp.close()
#
rootgrp = Dataset('urban.nc', 'r', format='NETCDF4')
frac_urban = np.flipud(rootgrp.variables['frac'][0][:][:])
rootgrp.close()
#
# coarser lat/lon grid (reduce by 6)
lons_lu, lats_lu = np.meshgrid(lon_lu[::6], lat_lu[::6])
#
crop = basemap.interp(frac_crop, lon_lu, lat_lu, lons_lu, lats_lu, order=0)
unused = basemap.interp(frac_unused, lon_lu, lat_lu, lons_lu, lats_lu, order=0)
forestry = basemap.interp(frac_forestry, lon_lu, lat_lu, lons_lu, lats_lu, order=0)
grazing = basemap.interp(frac_grazing, lon_lu, lat_lu, lons_lu, lats_lu, order=0)
urban = basemap.interp(frac_urban, lon_lu, lat_lu, lons_lu, lats_lu, order=0)
#
used = crop+grazing+forestry+urban
#
del frac_crop, frac_unused, frac_forestry, frac_urban, lon_lu, lat_lu
#
### calculating natural vs. managed dominant landuse
#
natvman = (unused) > (used) # used=0, unused=1
natvman = natvman*1


# In[15]:

###########################################################################
### loading MODIS biome data
landfilename = "MCD12C1.A2012001.051.2013178154403.nc"
print(' opening file: '+landfilename)
landfile = Dataset(landfilename, format='NETCDF4')
landlats = landfile.variables['latitude'][:]
landlons = landfile.variables['longitude'][:]
landmap_in = landfile.variables['Majority_Land_Cover_Type_1'][:]
landfile.close()
#
landlats_fix = flip(landlats,0)
landlons_fix = landlons
landmap_in_fix = np.flipud(landmap_in)
#
landmap_halfdegree = basemap.interp(landmap_in_fix, landlons_fix, landlats_fix, nlons, nlats, order=0)
#
del landlats_fix, landlons_fix, landlats, landlons, landmap_in
#
# initializing for grouping
landcover = landmap_halfdegree*1
landcover_all = landmap_halfdegree*1
old = landmap_halfdegree*1 # dummy variable
#
# excluding tundra, desert, peatland (1 to 7 categories)
landcover[((old == 1) | (old == 2) | (old == 3) | (old == 4) | (old == 5) | (old == 8)) #forests boreal
          & (nlats > 50)] = 1 
landcover[((old == 1) | (old == 2) | (old == 3) | (old == 4) | (old == 5) | (old == 8)) #forests temperate
          & (nlats < 50) & (nlats > 23)] = 2 
landcover[((old == 1) | (old == 2) | (old == 3) | (old == 4) | (old == 5) | (old == 8)) #forests temperate
          & (nlats > -50) & (nlats < -23)] = 2 
landcover[((old == 1) | (old == 2) | (old == 3) | (old == 4) | (old == 5) | (old == 8)) #forests tropical
          & (nlats < 23) & (nlats > -23)] = 3 
landcover[(old == 10) & (nlats < 60)] = 4 #grasslands
landcover[((old == 6) | (old == 7)) & (nlats < 60)] = 5 #shrublands
landcover[(old == 9)] = 6 #savannas
landcover[((old == 12) | (old == 13) | (old == 14))] = 7 #cropland
landcover[(old == 11)] = 0 #wetland/peatland
landcover[(old == 15)] = 0 #snow/ice
landcover[(old == 16)] = 0 #desert
landcover[((old == 0) | (old == 17))] = 0 #water and unclassified
landcover[((old == 6) | (old == 7)) & (nlats > 60)] = 0 #tundra shrubland
landcover[(old == 10) & (nlats > 60)] = 0 #tundra grassland
#
# including tundra, desert, peatland (1 to 10 categories, as in Shi et al. 2020)
landcover_all[((old == 1) | (old == 2) | (old == 3) | (old == 4) | (old == 5) | (old == 8)) #forests boreal
          & (nlats > 50)] = 1 
landcover_all[((old == 1) | (old == 2) | (old == 3) | (old == 4) | (old == 5) | (old == 8)) #forests temperate
          & (nlats < 50) & (nlats > 23)] = 2 
landcover[((old == 1) | (old == 2) | (old == 3) | (old == 4) | (old == 5) | (old == 8)) #forests temperate
          & (nlats > -50) & (nlats < -23)] = 2 
landcover_all[((old == 1) | (old == 2) | (old == 3) | (old == 4) | (old == 5) | (old == 8)) #forests tropical
          & (nlats < 23) & (nlats > -23)] = 3 
landcover_all[(old == 10) & (nlats < 60)] = 4 #grasslands
landcover_all[((old == 6) | (old == 7)) & (nlats < 60)] = 5 #shrublands
landcover_all[(old == 9)] = 6 #savannas
landcover_all[((old == 12) | (old == 13) | (old == 14))] = 7 #cropland
landcover_all[(old == 11)] = 10 #wetland/peatland
landcover_all[(old == 15)] = 0 #snow/ice
landcover_all[(old == 16)] = 9 #desert
landcover_all[((old == 0) | (old == 17))] = 0 #water and unclassified
landcover_all[((old == 6) | (old == 7)) & (nlats > 60)] = 8 #tundra shrubland
landcover_all[(old == 10) & (nlats > 60)] = 8 #tundra shrubland
#
# cleaning
del old, landmap_halfdegree


# In[ ]:

###########################################################################
### HARMONIZING AND PLOTTING
###########################################################################


# In[ ]:

# main variables of interest: 
# nlons, nlats
# cru_data, gpcc_data
# soc_t, soc_s, bd_t, bd_s
# texture_t, texture_s, activity_t, activity_s
# crop, unused, forestry, grazing, urban, used, landcover


# In[16]:

###########################################################################
### specify boundaries and ranges for figures
temp_levels = np.arange(-22,30,1)
rain_colorlevels=np.arange(0, 2000., 100)
texturelevels=np.arange(0,105,5)


# In[17]:

###########################################################################
### common masks between datasets
common_mask = np.logical_or(cru_data.mask[:], gpcc_data.mask[:])
common_mask = np.logical_or(common_mask[:], texture_t.mask[:])
common_mask = np.logical_or(common_mask[:], texture_s.mask[:])
#
soc_t_hwsd = np.ma.masked_array(soc_t_hwsd, mask=common_mask)
soc_s_hwsd = np.ma.masked_array(soc_s_hwsd, mask=common_mask)
soc_t_soilgrids = np.ma.masked_array(soc_t_soilgrids, mask=common_mask)
soc_s_soilgrids = np.ma.masked_array(soc_s_soilgrids, mask=common_mask)
cru_data = np.ma.masked_array(cru_data, mask=common_mask)
gpcc_data = np.ma.masked_array(gpcc_data, mask=common_mask)
texture_t = np.ma.masked_array(texture_t, mask=common_mask)
texture_s = np.ma.masked_array(texture_s, mask=common_mask)
activity_t = np.ma.masked_array(activity_t, mask=common_mask)
activity_s = np.ma.masked_array(activity_s, mask=common_mask)
#
natvman = np.ma.masked_array(natvman, mask=common_mask)
landcover = np.ma.masked_array(landcover, mask=common_mask) #no mask for landcover_all


# In[18]:

# map of soc (average of HWSD and Soilgrids)
soc_t_avg = (soc_t_hwsd + soc_t_soilgrids)/2
soc_s_avg = (soc_s_hwsd + soc_s_soilgrids)/2


# In[19]:

###########################################################################
### plotting key inputs
#
# plotting MAT
fig = plt.figure(figsize=(7,3))
ax = fig.add_axes([0.05,0.05,0.9,0.9])
m = Basemap(projection='gall',llcrnrlat=-60.5,urcrnrlat=90,             llcrnrlon=-170,urcrnrlon=190,resolution='c') 
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
bounds = temp_levels
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
cmap = plt.get_cmap('Spectral_r')
cmap.set_under(cmap(1))
im1 = m.pcolormesh(nlons, nlats, cru_data, norm=norm, shading='flat', cmap=cmap, latlon=True)
cb = m.colorbar(im1, size="3%", pad="4%",extend="both")
cb.set_ticks(bounds[::4])
cb.ax.tick_params(labelsize=14)
cb.set_label('MAT (\u00B0C)', fontsize=16)
ax.set_title('', fontsize=18) 
plt.show()
#
# plotting MAP
fig = plt.figure(figsize=(7,3))
ax = fig.add_axes([0.05,0.05,0.9,0.9])
m = Basemap(projection='gall',llcrnrlat=-60.5,urcrnrlat=90,             llcrnrlon=-170,urcrnrlon=190,resolution='c') 
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
bounds = rain_colorlevels
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
cmap = plt.get_cmap('Spectral_r')
cmap.set_under('white')
im1 = m.pcolormesh(nlons, nlats, gpcc_data, norm=norm, shading='flat', cmap=cmap, latlon=True)
cb = m.colorbar(im1, size="3%", pad="4%",extend="max")
cb.set_ticks(bounds[::5])
cb.ax.tick_params(labelsize=14)
cb.set_label('MAP (mm yr$^{-1}$)', fontsize=16)
ax.set_title('', fontsize=18) 
plt.show()
#
# plotting texture
fig = plt.figure(figsize=(7,3))
ax = fig.add_axes([0.05,0.05,0.9,0.9])
m = Basemap(projection='gall',llcrnrlat=-60.5,urcrnrlat=90,             llcrnrlon=-170,urcrnrlon=190,resolution='c')  
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
bounds = texturelevels
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
cmap = plt.get_cmap('viridis')
cmap.set_under('white')
im1 = m.pcolormesh(nlons, nlats, texture_t, norm=norm, shading='flat', vmin=0, vmax=100, cmap=cmap, latlon=True)
cb = m.colorbar(im1, size="3%", pad="4%")
cb.set_ticks(bounds[::2])
cb.ax.tick_params(labelsize=14)
cb.set_label('Clay + Silt content (%)', fontsize=16)
ax.set_title('Topsoil (<30 cm)', fontsize=12) 
plt.show()
#
# plotting clay activity
fig = plt.figure(figsize=(7,3))
ax = fig.add_axes([0.05,0.05,0.9,0.9])
m = Basemap(projection='gall',llcrnrlat=-60.5,urcrnrlat=90,             llcrnrlon=-170,urcrnrlon=190,resolution='c') 
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
cmap = cm.get_cmap('Accent_r', 2) 
cmap.set_under('white')
im1 = m.pcolormesh(nlons, nlats, activity_t, shading='flat', vmin = 0, vmax = 1, cmap=cmap, latlon=True)
cb = m.colorbar(im1, size="3%", pad="4%")
cb.set_ticks([0,1])
cb.set_ticklabels(['Low-activity \n minerals','High-activity \n minerals'])
cb.ax.tick_params(labelsize=12)
ax.set_title('Topsoil (<30 cm)', fontsize=12) 
plt.show()


# In[20]:

###########################################################################
### plotting soc
cmap = matplotlib.cm.get_cmap('Spectral')
rgba = cmap(0)
#
cmap = plt.get_cmap('Spectral_r')
cmap.set_under('white')
cmap.set_over(rgba)
bounds = np.array([0, 0.1, 1, 2, 5, 10, 20, 50, 100])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
#
fig = plt.figure(figsize=(7,3))
ax = fig.add_axes([0.05,0.05,0.9,0.9])
m = Basemap(projection='gall',llcrnrlat=-60.5,urcrnrlat=90,             llcrnrlon=-170,urcrnrlon=190,resolution='c') 
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
im1 = m.pcolormesh(nlons, nlats, soc_t_avg, vmin = 0, norm=norm, shading='flat', cmap=cmap, latlon=True)
cb = m.colorbar(im1, size="3%", pad="4%", extend="max")
cb.ax.tick_params(labelsize=14)
cb.set_label('SOC (g C kg$^{-1}$ soil)', fontsize=16)
ax.set_title('Topsoil (<30 cm)', fontsize=12) 
plt.show()


# In[21]:

###########################################################################
### plotting fractional land-use
fig = plt.figure(figsize=(15,9))
plt.subplot(221)
#
cmap = matplotlib.cm.get_cmap('BuGn_r')
rgba = cmap(0)
#
cmap = plt.get_cmap('BuGn')
cmap.set_over(rgba)
cmap.set_under('white')
#
m = Basemap(projection='gall',llcrnrlat=-60.5,urcrnrlat=90,             llcrnrlon=-170,urcrnrlon=190,resolution='c') 
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
im1 = m.pcolormesh(nlons, nlats, crop, shading='flat', vmin = 0, vmax = 1, cmap=cmap, latlon=True)
cb = m.colorbar(im1, size="5%", pad="6%", location='bottom')
cb.ax.tick_params(labelsize=14)
cb.set_label('Fraction of gridcell with crop lands', fontsize=14)
plt.annotate('Crop land', xy=(0.02, 0.05), fontsize=12, xycoords='axes fraction')
#
plt.subplot(222)
#
m = Basemap(projection='gall',llcrnrlat=-60.5,urcrnrlat=90,             llcrnrlon=-170,urcrnrlon=190,resolution='c') 
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
im1 = m.pcolormesh(nlons, nlats, grazing, shading='flat', vmin = 0, vmax = 1, cmap=cmap, latlon=True)
cb = m.colorbar(im1, size="5%", pad="6%", location='bottom')
cb.ax.tick_params(labelsize=14)
cb.set_label('Fraction of gridcell with grazing lands', fontsize=14)
plt.annotate('Grazing land', xy=(0.02, 0.05), fontsize=12, xycoords='axes fraction')
#
plt.subplot(223)
#
m = Basemap(projection='gall',llcrnrlat=-60.5,urcrnrlat=90,             llcrnrlon=-170,urcrnrlon=190,resolution='c') 
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
im1 = m.pcolormesh(nlons, nlats, forestry, shading='flat', vmin = 0, vmax = 1, cmap=cmap, latlon=True)
cb = m.colorbar(im1, size="5%", pad="6%", location='bottom')
cb.ax.tick_params(labelsize=14)
cb.set_label('Fraction of gridcell with forestry lands', fontsize=14)
plt.annotate('Forestry land', xy=(0.02, 0.05), fontsize=12, xycoords='axes fraction')
#
plt.subplot(224)
#
m = Basemap(projection='gall',llcrnrlat=-60.5,urcrnrlat=90,             llcrnrlon=-170,urcrnrlon=190,resolution='c') 
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
im1 = m.pcolormesh(nlons, nlats, (1-unused), shading='flat', vmin = 0, vmax = 1, cmap=cmap, latlon=True)
cb = m.colorbar(im1, size="5%", pad="6%", location='bottom')
cb.ax.tick_params(labelsize=14)
cb.set_label('Fraction of gridcell with \n grazing, crop, forestry lands', fontsize=14)
plt.annotate('All used land', xy=(0.02, 0.05), fontsize=12, xycoords='axes fraction')
plt.subplots_adjust(wspace=0.1, hspace=0.2)
plt.show()


# In[22]:

###########################################################################
### plotting all landcover type
fig = plt.figure(figsize=(7,3))
ax = fig.add_axes([0.05,0.05,0.9,0.9])
m = Basemap(projection='gall',llcrnrlat=-60.5,urcrnrlat=90,             llcrnrlon=-170,urcrnrlon=190,resolution='c')  
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
cmap = plt.get_cmap('tab10',10)
cmap.set_under('white')
im1 = m.pcolormesh(nlons, nlats, landcover_all, shading='flat', vmin = 0.5, vmax = 10.5, cmap=cmap, latlon=True)
cb = m.colorbar(im1, size="3%", pad="4%")
cb.set_ticks([1,2,3,4,5,6,7,8,9,10])
cb.set_ticklabels(['Boreal Forest','Temperate Forest','Tropical Forest','Grassland',
                   'Shrubland','Savanna','Cropland','Tundra','Desert','Peatland'])
cb.ax.tick_params(labelsize=12)
#cb.set_label('PFT category (0 to 18)', fontsize=14)
ax.set_title('', fontsize=18) 
plt.show() 


# In[23]:

###########################################################################
### plotting organic soils
fig = plt.figure(figsize=(7,3))
ax = fig.add_axes([0.05,0.05,0.9,0.9])
m = Basemap(projection='gall',llcrnrlat=-60.5,urcrnrlat=90,             llcrnrlon=-170,urcrnrlon=190,resolution='c') 
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
cmap = plt.get_cmap('Greys')
cmap.set_under('white')
im1 = m.pcolormesh(nlons, nlats, organics_commonmap, shading='flat', cmap=cmap, latlon=True)
cb = m.colorbar(im1, size="5%", pad="10%")
cb.set_ticks([0, 1])
cb.ax.tick_params(labelsize=14)
cb.set_label('Organic soil (>50%)', fontsize=16) # vs. \n presence/absence
ax.set_title('', fontsize=18) 
plt.show()


# In[24]:

###########################################################################
# adding peatland mask to all
common_mask = np.logical_or(common_mask[:], organics_commonmap)
#
soc_t_hwsd = np.ma.masked_array(soc_t_hwsd, mask=common_mask)
soc_s_hwsd = np.ma.masked_array(soc_s_hwsd, mask=common_mask)
soc_t_soilgrids = np.ma.masked_array(soc_t_soilgrids, mask=common_mask)
soc_s_soilgrids = np.ma.masked_array(soc_s_soilgrids, mask=common_mask)
soc_t_avg = np.ma.masked_array(soc_t_avg, mask=common_mask)
soc_s_avg = np.ma.masked_array(soc_s_avg, mask=common_mask)
cru_data = np.ma.masked_array(cru_data, mask=common_mask)
gpcc_data = np.ma.masked_array(gpcc_data, mask=common_mask)
texture_t = np.ma.masked_array(texture_t, mask=common_mask)
texture_s = np.ma.masked_array(texture_s, mask=common_mask)
activity_t = np.ma.masked_array(activity_t, mask=common_mask)
activity_s = np.ma.masked_array(activity_s, mask=common_mask)
#
natvman = np.ma.masked_array(natvman, mask=common_mask)
landcover = np.ma.masked_array(landcover, mask=common_mask) #no mask for landcover_all


# In[ ]:

###########################################################################
### RF PREDICTIONS
###########################################################################


# In[25]:

# choosing SOC --> soc_t_hwsd, soc_t_soilgrids, soc_t_avg
soc_t = soc_t_avg
soc_s = soc_s_avg


# In[26]:

###########################################################################
### flattened global datasets
LU = landcover # choosing  landuse to match synthesis categories
#
ma.set_fill_value(soc_t, 0) 
ma.set_fill_value(texture_t, 0)
ma.set_fill_value(cru_data, 0)
ma.set_fill_value(gpcc_data, 0)
ma.set_fill_value(LU, 0)
ma.set_fill_value(activity_t, 0)
#
length = np.ravel((soc_t)).shape[0] 
som = np.ravel((soc_t)).reshape(length, 1) 
tex = np.ravel((texture_t)).reshape(length, 1)
mat = np.ravel((cru_data)).reshape(length, 1)
precip = np.ravel((gpcc_data)).reshape(length, 1)
veg = np.ravel((LU)).reshape(length, 1) 
mineral = np.ravel((activity_t)).reshape(length, 1)
#
somtex_t = np.asarray(np.concatenate((som, tex, mat, precip, veg, mineral), axis=1))
#
ma.set_fill_value(soc_s, 0) 
ma.set_fill_value(texture_s, 0)
ma.set_fill_value(activity_s, 0)
#
length = np.ravel((soc_s)).shape[0] 
som = np.ravel((soc_s)).reshape(length, 1) 
tex = np.ravel((texture_s)).reshape(length, 1)
mat = np.ravel((cru_data)).reshape(length, 1)
precip = np.ravel((gpcc_data)).reshape(length, 1)
veg = np.ravel((LU)).reshape(length, 1)
mineral = np.ravel((activity_s)).reshape(length, 1)
#
somtex_s = np.asarray(np.concatenate((som, tex, mat, precip, veg, mineral), axis=1))
#
del length, som, tex, mat, precip, veg, mineral


# In[27]:

###########################################################################
### loading training data
dataglobal = pd.read_csv("rf-data.csv") ## from synthesis data (MOC_synthesis.csv) with non-NA values for Bulk.C, SiltClayPercent, MAT, MAP, Vegetation, MineralType, SiltClayC, POM_C
dataglobal = np.array(dataglobal) # columns = SOC, TEX, MAT, MAP, veg, MineralType, MOC, POC
#
mocindex=6
print(dataglobal.shape)


# In[28]:

###########################################################################
### plotting histograms of SOC
fig = plt.figure(figsize=(16,8))
plt.subplot(231)
plt.hist(dataglobal[:,0], bins=30, normed=True, alpha=0.5, 
               histtype='stepfilled', color='green', edgecolor='none', label='Synthesis Data')
#
plt.hist(np.ravel(soc_s_hwsd[ (soc_s_hwsd > 0) & (soc_s_hwsd < 120) ]), bins=30, normed=True, alpha=0.5,
         histtype='stepfilled', color='red', edgecolor='none', label='Global Subsoil')
#
plt.hist(np.ravel(soc_t_hwsd[ (soc_t_hwsd > 0) & (soc_t_hwsd < 120) ]), bins=30, normed=True, alpha=0.75,
         histtype='stepfilled', color='steelblue', edgecolor='none', label='Global Topsoil')
#
plt.xlabel('SOC (g C kg$^{-1}$ soil)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tick_params(labelsize=14)
plt.title('HWSD')
##
plt.subplot(232)
plt.hist(dataglobal[:,0], bins=30, normed=True, alpha=0.5, 
               histtype='stepfilled', color='green', edgecolor='none', label='Synthesis Data')
#
plt.hist(np.ravel(soc_s_avg[ (soc_s_avg > 0) & (soc_s_avg < 120) ]), bins=30, normed=True, alpha=0.5,
         histtype='stepfilled', color='red', edgecolor='none', label='Global Subsoil')
#
plt.hist(np.ravel(soc_t_avg[ (soc_t_avg > 0) & (soc_t_avg < 120) ]), bins=30, normed=True, alpha=0.75,
         histtype='stepfilled', color='steelblue', edgecolor='none', label='Global Topsoil')
#
plt.xlabel('SOC (g C kg$^{-1}$ soil)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tick_params(labelsize=14)
plt.title('Averaged Data Product')
##
plt.subplot(233)
plt.hist(dataglobal[:,0], bins=30, normed=True, alpha=0.5, 
               histtype='stepfilled', color='green', edgecolor='none', label='Synthesis Data')
#
plt.hist(np.ravel(soc_s_soilgrids[ (soc_s_soilgrids > 0) & (soc_s_soilgrids < 120) ]), bins=30, normed=True, alpha=0.5,
         histtype='stepfilled', color='red', edgecolor='none', label='Global Subsoil')
#
plt.hist(np.ravel(soc_t_soilgrids[ (soc_t_soilgrids > 0) & (soc_t_soilgrids < 120) ]), bins=30, normed=True, alpha=0.75,
         histtype='stepfilled', color='steelblue', edgecolor='none', label='Global Topsoil')
#
plt.xlabel('SOC (g C kg$^{-1}$ soil)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tick_params(labelsize=14)
plt.title('SoilGrids')
plt.legend(bbox_to_anchor=(0.4, 0.95), loc='upper left', borderaxespad=0., fontsize=13)
#
plt.show()


# In[29]:

###########################################################################
### RF for MOC
X_train, X_test, y_train, y_test = train_test_split(dataglobal[:,0:mocindex], dataglobal[:,mocindex], test_size=0.25, random_state=0)
#
clf = RandomForestRegressor(n_estimators=400)
clf = clf.fit(X_train, y_train)
#
Ypredicted = clf.predict(X_test)
#
r2 = r2_score(y_test, Ypredicted)
print("R2 = ", r2)
#
fig = plt.scatter(y_test, Ypredicted, marker='.', color = 'k', s=200)
plt.axis([0, 80, 0, 80])
plt.xlabel('Observed MOC (g C kg$^{-1}$ soil)', fontsize=16)
plt.ylabel('Predicted MOC (g C kg$^{-1}$ soil)', fontsize=16)
plt.plot([0, 100], [0, 100], 'k')
plt.tick_params(labelsize=14)
#
print("Accuracy on training set: {:.3f}".format(clf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(clf.score(X_test, y_test)))
#
err_down, err_up = pred_ints(clf, X_test, percentile=90)
#
fig = plt.errorbar(y_test, Ypredicted, yerr = [np.absolute(Ypredicted-err_down), np.absolute(err_up-Ypredicted)],
                  fmt='o', mfc='k', mec='k', ms=8, ecolor='grey', capthick=2)
plt.axis([0, 80, 0, 80])
plt.xlabel('Observed MOC (g C kg$^{-1}$ soil)', fontsize=16)
plt.ylabel('Predicted MOC (g C kg$^{-1}$ soil)', fontsize=16)
plt.plot([0, 100], [0, 100], 'k')
plt.tick_params(labelsize=14)


# In[30]:

###########################################################################
### Ensemble RF for MOC
R2trainall = []
R2testall = []
#
for i in range(300):
    X_train, X_test, y_train, y_test = train_test_split(dataglobal[:,0:mocindex], dataglobal[:,mocindex], test_size=0.25, random_state=i)
    #
    ensembleclf = RandomForestRegressor(n_estimators=400)
    ensembleclf = ensembleclf.fit(X_train, y_train)
    #
    R2train = ensembleclf.score(X_train, y_train)
    R2test = ensembleclf.score(X_test, y_test)
    #
    R2trainall.append(R2train)
    R2testall.append(R2test)


# In[31]:

print('Global predictions: R2=', np.round(np.mean(R2testall),3), '+/-', np.round(np.std(R2testall),3))
#
n = np.arange(300)
#
fig = plt.figure(figsize=(5,4))
plt.fill_between([0, 300], 
                  np.round(np.mean(R2testall),2)-np.round(np.std(R2testall),2), 
                  np.round(np.mean(R2testall),2)+np.round(np.std(R2testall),2), color='grey', alpha=0.5)
scatter(n, R2testall, marker='o', color = 'k', s=20, label='Global predictions: f(MAT, MAP, CS, SOC, veg, mineral)')
plt.axis([0, 300, 0, 1])
plt.plot([0, 300], [np.round(np.mean(R2testall),2), np.round(np.mean(R2testall),2)], 'k--')
plt.xlabel('Iteration (seed number)', fontsize=16)
plt.ylabel('Test R$^{2}$', fontsize=16)
plt.tick_params(labelsize=14)
plt.show()


# In[32]:

###########################################################################
### Predicting MOC globally
MOCpred_t = clf.predict(somtex_t[:,0:mocindex])
MOCpred_s = clf.predict(somtex_s[:,0:mocindex])
#
### Reshaping MOC to grid
MOCpred_map_t = MOCpred_t.reshape(soc_t.shape[0], soc_t.shape[1])
MOCpred_map_s = MOCpred_s.reshape(soc_t.shape[0], soc_t.shape[1])

# In[33]:

###########################################################################
### Fixing masks
MOCpred_map_t = ma.masked_where(MOCpred_map_t == MOCpred_t[0], MOCpred_map_t)
MOCpred_map_s = ma.masked_where(MOCpred_map_s == MOCpred_s[0], MOCpred_map_s)
#
common_mask = np.logical_or(soc_t.mask[:], MOCpred_map_t.mask[:])
MOCpred_map_t = np.ma.masked_array(MOCpred_map_t, mask=common_mask)
MOCpred_map_s = np.ma.masked_array(MOCpred_map_s, mask=common_mask)


# In[ ]:

###########################################################################
### MOCmax calculations & converting all predictions to stocks
###########################################################################


# In[34]:

###########################################################################
### Calculating MOCmax for surface soils
depth = 30 # in cm
#
BD = bd_t
activity = activity_t
texture = texture_t
#
slope11 = 0.48 ## derived from MOC synthesis for 1:1 clays
slope21 = 0.86 ## derived from MOC synthesis for 2:1 clays
slopemap = activity*slope21 + (1-activity)*slope11
#
Qmax_gkg_t = (texture*slopemap) # using slope from synthesis for max MOC (in gC/kg soil) 
potential_MOC_kgm2_t = (Qmax_gkg_t/10)*BD*(depth/10) # converting with BD and depth  
#
current_MOC_kgm2_t = (MOCpred_map_t/10)*BD*(depth/10)
#
soc_kgm2_t = (soc_t/10)*BD*(depth/10) # soc_t in gC/kg soil


# In[35]:

###########################################################################
### Calculating MOCmax for deep soils
depth = 70 # in cm
#
BD = bd_s
activity = activity_s
texture = texture_s
#
slopemap = activity*slope21 + (1-activity)*slope11
#
Qmax_gkg_s = (texture*slopemap) # using slope from synthesis for max MOC (in gC/kg soil)
potential_MOC_kgm2_s = (Qmax_gkg_s/10)*BD*(depth/10)  
#
current_MOC_kgm2_s = (MOCpred_map_s/10)*BD*(depth/10)
#
soc_kgm2_s = (soc_s/10)*BD*(depth/10) # soc_s in gC/kg soil


# In[36]:

###########################################################################
### Calculating MOCmax for all depths
MOCmax_kgm2 = potential_MOC_kgm2_t + potential_MOC_kgm2_s
#
MOC_kgm2 = current_MOC_kgm2_t + current_MOC_kgm2_s
#
SOC_kgm2 = soc_kgm2_t + soc_kgm2_s


# In[37]:

###########################################################################
### MOCmax summary
cmap = matplotlib.cm.get_cmap('BrBG')
rgba = cmap(0)
#
bounds = np.array([0, 1, 2, 5, 10, 20, 50, 80, 100]) 
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
cmap = plt.get_cmap('BrBG_r')
rgba_under = cmap(0)
cmap.set_under('white')
cmap.set_over(rgba)
#
fig = plt.figure(figsize=(7,3))
ax = fig.add_axes([0.05,0.05,0.9,0.9])
m = Basemap(projection='gall',llcrnrlat=-62,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,resolution='c')  
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
im1 = m.pcolormesh(nlons, nlats, potential_MOC_kgm2_t*(landcover>0), shading='flat', norm=norm, cmap=cmap, latlon=True)
cb = m.colorbar(im1, size="3%", pad="4%", extend="max")
cb.ax.tick_params(labelsize=14)
cb.set_label('MOC$_{max}$ (kg C m$^{-2}$ soil)', fontsize=16)
ax.set_title('Topsoil (<30 cm)', fontsize=12)
cb.ax.set_yticklabels(['0', '1', '2', '5', '10', '20', '50', '80', '100'])  
plt.show()
#
fig = plt.figure(figsize=(7,3))
ax = fig.add_axes([0.05,0.05,0.9,0.9])
m = Basemap(projection='gall',llcrnrlat=-62,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,resolution='c')  
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
im1 = m.pcolormesh(nlons, nlats, potential_MOC_kgm2_s*(landcover>0), shading='flat', norm=norm, cmap=cmap, latlon=True)
cb = m.colorbar(im1, size="3%", pad="4%", extend="max")
cb.ax.tick_params(labelsize=14)
cb.set_label('MOC$_{max}$ (kg C m$^{-2}$ soil)', fontsize=16)
ax.set_title('Subsoil (30-100 cm)', fontsize=12) 
cb.ax.set_yticklabels(['0', '1', '2', '5', '10', '20', '50', '80', '100'])
plt.show()


# In[38]:

###########################################################################
### MOC summary
bounds = np.array([0, 0.1, 1, 2, 5, 10, 20, 50])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
cmap = plt.get_cmap('RdYlBu_r')
cmap.set_under('white')
#
fig = plt.figure(figsize=(7,4))
ax = fig.add_axes([0.05,0.05,0.9,0.9])
m = Basemap(projection='gall',llcrnrlat=-62,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,resolution='c')  
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
im1 = m.pcolormesh(nlons, nlats, current_MOC_kgm2_t*(landcover>0), norm=norm, shading='flat', cmap=cmap, latlon=True)
cb = m.colorbar(im1, size="3%", pad="4%", extend="max")
cb.ax.tick_params(labelsize=14)
cb.set_label('MOC (kg C m$^{-2}$ soil)', fontsize=16)
ax.set_title('Topsoil (<30 cm)', fontsize=12) 
cb.ax.set_xticklabels(['0', '0.1', '1', '2', '5', '10', '20', '50'])
plt.show()
#
fig = plt.figure(figsize=(7,4))
ax = fig.add_axes([0.05,0.05,0.9,0.9])
m = Basemap(projection='gall',llcrnrlat=-62,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,resolution='c')  
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
im1 = m.pcolormesh(nlons, nlats, current_MOC_kgm2_s*(landcover>0), norm=norm, shading='flat', cmap=cmap, latlon=True)
cb = m.colorbar(im1, size="3%", pad="4%", extend="max")
cb.ax.tick_params(labelsize=14)
cb.set_label('MOC (kg C m$^{-2}$ soil)', fontsize=16)
ax.set_title('Subsoil (30-100 cm)', fontsize=12) 
cb.ax.set_xticklabels(['0', '0.1', '1', '2', '5', '10', '20', '50'])
plt.show()


# In[39]:

###########################################################################
### Calculating MOC/SOC
MOCtoSOC_t = current_MOC_kgm2_t/soc_kgm2_t
MOCtoSOC_s = current_MOC_kgm2_s/soc_kgm2_s
MOCtoSOC_1m = (current_MOC_kgm2_t+current_MOC_kgm2_s)/(soc_kgm2_t+soc_kgm2_s)
#
print('Over represented landcovers: \n')
print('MOC/SOC topsoils:', round(np.mean(MOCtoSOC_t[ (MOCtoSOC_t > 0) & (landcover > 0) ]),2), '(+/-)', round(np.std(MOCtoSOC_t[ (MOCtoSOC_t > 0) & (landcover > 0) ]),2))
#
print('MOC/SOC subsoils:', round(np.mean(MOCtoSOC_s[ (MOCtoSOC_s > 0) & (landcover > 0) ]),2), '(+/-)', round(np.std(MOCtoSOC_s[ (MOCtoSOC_s > 0) & (landcover > 0) ]),2))
#
print('MOC/SOC all depths:', round(np.mean(MOCtoSOC_1m[ (MOCtoSOC_1m > 0) & (landcover > 0) ]),2), '(+/-)', round(np.std(MOCtoSOC_1m[ (MOCtoSOC_1m > 0) & (landcover > 0) ]),2))
#
print('Data all depths:', round(np.mean(100*dataglobal[:,mocindex]/dataglobal[:,0]),1))  


# In[40]:

###########################################################################
### Calculating percent MOC saturation
percentsat_t = current_MOC_kgm2_t/potential_MOC_kgm2_t*100
percentsat_t = np.ma.masked_array(percentsat_t, mask=common_mask)
#
deficit_MOC_kgm2_t = potential_MOC_kgm2_t-current_MOC_kgm2_t
#
### Calculating percent MOC saturation for subsoil
percentsat_s = current_MOC_kgm2_s/potential_MOC_kgm2_s*100
percentsat_s = np.ma.masked_array(percentsat_s, mask=common_mask)
#
deficit_MOC_kgm2_s = potential_MOC_kgm2_s-current_MOC_kgm2_s
#
### Calculating percent MOC saturation for all depths
percentsat_1m = (current_MOC_kgm2_t+current_MOC_kgm2_s)/(potential_MOC_kgm2_t+potential_MOC_kgm2_s)*100
percentsat_1m = np.ma.masked_array(percentsat_1m, mask=common_mask)


# In[41]:

print('Represented biomes:') # adding mask for non-permafrost, non-desert soils
print('Topsoil mean %Csat =', round(np.mean(percentsat_t[ (percentsat_t > 0) & (landcover > 0) ]),1))
print('Subsoil mean %Csat =', round(np.mean(percentsat_s[ (percentsat_s > 0) & (landcover > 0) ]),1))
print('All depths mean %Csat =', round(np.mean(percentsat_1m[ (percentsat_1m > 0) & (landcover > 0) ]),1))
#
data_sat = 100*dataglobal[:,mocindex]/(dataglobal[:,1]*0.86) #for quick comparison
print('Data mean %Csat =', round(np.mean(data_sat),1))  


# In[42]:

###########################################################################
### MOC saturation summary
fig = plt.figure(figsize=(6,4))
plt.hist(data_sat[ (data_sat > 0) & (data_sat < 100)], bins=30, normed=True, alpha=0.5,
         histtype='stepfilled', color='green',
         edgecolor='none', label='Synthesis Data')
#
plt.hist(np.ravel(percentsat_s[ (percentsat_s > 0) & (percentsat_s < 100) & (landcover >0) ]), bins=30, normed=True, alpha=0.5,
         histtype='stepfilled', color='red',
         edgecolor='none', label='Global Subsoil')
#
plt.hist(np.ravel(percentsat_t[ (percentsat_t > 0) & (percentsat_t < 100) & (landcover >0) ]), bins=30, normed=True, alpha=0.75,
         histtype='stepfilled', color='steelblue',
         edgecolor='none', label='Global Topsoil')
plt.xlabel('%C saturation', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend(bbox_to_anchor=(0.55, 0.95), loc='upper left', borderaxespad=0., fontsize=13)
plt.tick_params(labelsize=14)
plt.show()


# In[43]:

cmap = matplotlib.cm.get_cmap('Spectral_r')
rgba = cmap(0)
#
cmap = plt.get_cmap('Spectral')
cmap.set_over(rgba)
cmap.set_under('white')
#
fig = plt.figure(figsize=(7,4))
ax = fig.add_axes([0.05,0.05,0.9,0.9])
m = Basemap(projection='gall',llcrnrlat=-62,urcrnrlat=90,             llcrnrlon=-180,urcrnrlon=180,resolution='c') 
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
im1 = m.pcolormesh(nlons, nlats, percentsat_t*(landcover>0), vmin = 0, shading='flat', vmax = 100, cmap=cmap, latlon=True)
cb = m.colorbar(im1, size="3%", pad="4%")
cb.ax.tick_params(labelsize=14)
cb.set_label('%C saturation', fontsize=16)
ax.set_title('Topsoil (<30 cm)', fontsize=12) 
plt.show()
#
fig = plt.figure(figsize=(7,4))
ax = fig.add_axes([0.05,0.05,0.9,0.9])
m = Basemap(projection='gall',llcrnrlat=-62,urcrnrlat=90,             llcrnrlon=-180,urcrnrlon=180,resolution='c') 
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
im1 = m.pcolormesh(nlons, nlats, percentsat_s*(landcover>0), vmin = 0, shading='flat', vmax = 100, cmap=cmap, latlon=True)
cb = m.colorbar(im1, size="3%", pad="4%")
cb.ax.tick_params(labelsize=14)
cb.set_label('%C saturation', fontsize=16)
ax.set_title('Subsoil (30-100 cm)', fontsize=12) 
plt.show()


# In[44]:

cmap = matplotlib.cm.get_cmap('Spectral')
rgba = cmap(0)
#
cmap = plt.get_cmap('Spectral_r')
cmap.set_over(rgba)
cmap.set_under('white')
#
fig = plt.figure(figsize=(7,4))
ax = fig.add_axes([0.05,0.05,0.9,0.9])
m = Basemap(projection='gall',llcrnrlat=-62,urcrnrlat=90,             llcrnrlon=-180,urcrnrlon=180,resolution='c') 
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
im1 = m.pcolormesh(nlons, nlats, deficit_MOC_kgm2_t*(landcover>0), shading='flat', vmin = 0, vmax = 30, cmap=cmap, latlon=True)
cb = m.colorbar(im1, size="3%", pad="4%", extend="max")
cb.ax.tick_params(labelsize=14)
cb.set_label('C deficit (kg C m$^{-2}$ soil)', fontsize=16)
ax.set_title('Topsoil (<30 cm)', fontsize=12) 
plt.show()
#
fig = plt.figure(figsize=(7,4))
ax = fig.add_axes([0.05,0.05,0.9,0.9])
m = Basemap(projection='gall',llcrnrlat=-62,urcrnrlat=90,             llcrnrlon=-180,urcrnrlon=180,resolution='c') 
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
im1 = m.pcolormesh(nlons, nlats, deficit_MOC_kgm2_s*(landcover>0), shading='flat', vmin = 0, vmax = 70, cmap=cmap, latlon=True)
cb = m.colorbar(im1, size="3%", pad="4%", extend="max")
cb.ax.tick_params(labelsize=14)
cb.set_label('C deficit (kg C m$^{-2}$ soil)', fontsize=16)
ax.set_title('Subsoil (30-100 cm)', fontsize=12) 
plt.show()


# In[45]:

###########################################################################
### MOCmax, MOC, %C saturation main figure
fig = plt.figure(figsize=(15,9))
plt.subplot(221)
#
cmap = plt.get_cmap('BrBG_r')
cmap.set_under('white')
#
m = Basemap(projection='robin',lon_0=0,resolution='c')  
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
im1 = m.pcolormesh(nlons, nlats, potential_MOC_kgm2_t*(landcover>0), shading='flat', vmin=0, vmax=30, cmap=cmap, latlon=True)
cb = m.colorbar(im1, size="5%", pad="6%", extend='max', location='bottom')
cb.ax.tick_params(labelsize=14)
cb.set_label('MOC$_{max}$ (kg C m$^{-2}$ soil)', fontsize=15)
ax.set_title('Topsoil (<30 cm)', fontsize=12) 
ax.set_title('', fontsize=18) 
#
plt.subplot(222)
#
bounds = np.array([0, 0.1, 1, 2, 5, 10, 20, 30])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
cmap = plt.get_cmap('RdYlBu_r')
cmap.set_under('white')
#
m = Basemap(projection='robin',lon_0=0,resolution='c')  
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
im1 = m.pcolormesh(nlons, nlats, current_MOC_kgm2_t*(landcover>0), norm=norm, shading='flat', cmap=cmap, latlon=True)
cb = m.colorbar(im1, size="5%", pad="6%", extend='max', location='bottom')
cb.ax.tick_params(labelsize=14)
cb.set_label('MOC (kg C m$^{-2}$ soil)', fontsize=15)
ax.set_title('Topsoil (<30 cm)', fontsize=12) 
cb.ax.set_xticklabels(['0', '0.1', '1', '2', '5', '10', '20', '30'])
ax.set_title('', fontsize=18) 
#
plt.subplot(223)
#
cmap = matplotlib.cm.get_cmap('Spectral_r')
rgba = cmap(0)
#
cmap = plt.get_cmap('Spectral')
cmap.set_over(rgba)
cmap.set_under('white')
#
m = Basemap(projection='robin',lon_0=0,resolution='c') 
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
im1 = m.pcolormesh(nlons, nlats, percentsat_t*(landcover>0), shading='flat', vmin = 0, vmax = 50, cmap=cmap, latlon=True)
cb = m.colorbar(im1, size="5%", pad="6%", extend = 'max', location='bottom')
cb.ax.tick_params(labelsize=14)
cb.set_label('%C saturation', fontsize=15)
ax.set_title('Topsoil (<30 cm)', fontsize=12) 
#cb.ax.set_xticklabels(['0','10', '20', '30', '40', '50'])
plt.subplots_adjust(wspace=0.1, hspace=0.3)
plt.show()


# In[ ]:

###########################################################################
### Global totals
###########################################################################


# In[46]:

# calculating total stocks
print('On all soils:')
SOCPg_t = np.sum(soc_kgm2_t * area_ud)/1e12 # in Pg
SOCPg_s = np.sum(soc_kgm2_s * area_ud)/1e12 
totSOCPg = np.sum(SOC_kgm2 * area_ud)/1e12 
#
print('Topsoil SOC (PgC) = \t', round(SOCPg_t,1))  
print('Subsoil SOC (PgC) = \t', round(SOCPg_s,1))  
print('Total SOC (PgC) = \t', round(SOCPg_t+SOCPg_s,1))  
#
MOCPg_t = np.sum(current_MOC_kgm2_t * area_ud)/1e12 # in Pg
MOCPg_s = np.sum(current_MOC_kgm2_s * area_ud)/1e12 
totMOCPg = np.sum(MOC_kgm2 * area_ud)/1e12 
#
print('Topsoil MOC (PgC) = \t', round(MOCPg_t,1))  
print('Subsoil MOC (PgC) = \t', round(MOCPg_s,1))    
print('Total MOC (PgC) = \t', round(MOCPg_t+MOCPg_s,1))  
#
MOCmaxPg_t = np.sum(potential_MOC_kgm2_t * area_ud)/1e12 # in Pg 
MOCmaxPg_s = np.sum(potential_MOC_kgm2_s * area_ud)/1e12 
totMOCmaxPg = np.sum(MOCmax_kgm2 * area_ud)/1e12 
# 
print('Topsoil MOCmax (PgC) = \t', round(MOCmaxPg_t,1))  
print('Subsoil MOCmax (PgC) = \t', round(MOCmaxPg_s,1))    
print('Total MOCmax (PgC) = \t', round(MOCmaxPg_t+MOCmaxPg_s,1))    


# In[47]:

# calculating total stocks
print('On represented/non-permafrost soils:')
SOCPg_t = np.sum(soc_kgm2_t * (landcover>0) * area_ud)/1e12 # in Pg
SOCPg_s = np.sum(soc_kgm2_s * (landcover>0) * area_ud)/1e12 
totSOCPg = np.sum(SOC_kgm2 * (landcover>0) * area_ud)/1e12 
#
print('Topsoil SOC (PgC) = \t', round(SOCPg_t,1))  
print('Subsoil SOC (PgC) = \t', round(SOCPg_s,1))  
print('Total SOC (PgC) = \t', round(SOCPg_t+SOCPg_s,1))  
#
MOCPg_t = np.sum(current_MOC_kgm2_t * (landcover>0) * area_ud)/1e12 # in Pg
MOCPg_s = np.sum(current_MOC_kgm2_s * (landcover>0) * area_ud)/1e12 
totMOCPg = np.sum(MOC_kgm2 * (landcover>0) * area_ud)/1e12 
#
print('Topsoil MOC (PgC) = \t', round(MOCPg_t,1))  
print('Subsoil MOC (PgC) = \t', round(MOCPg_s,1))    
print('Total MOC (PgC) = \t', round(MOCPg_t+MOCPg_s,1))  
#
MOCmaxPg_t = np.sum(potential_MOC_kgm2_t * (landcover>0) * area_ud)/1e12 
MOCmaxPg_s = np.sum(potential_MOC_kgm2_s * (landcover>0) * area_ud)/1e12 
totMOCmaxPg = np.sum(MOCmax_kgm2 * (landcover>0) * area_ud)/1e12 
# 
print('Topsoil MOCmax (PgC) = \t', round(MOCmaxPg_t,1))  
print('Subsoil MOCmax (PgC) = \t', round(MOCmaxPg_s,1))    
print('Total MOCmax (PgC) = \t', round(MOCmaxPg_t+MOCmaxPg_s,1))    


# In[ ]:

###########################################################################
###########################################################################

