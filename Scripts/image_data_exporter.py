import sys
import warnings
import ee
from tqdm import tqdm
import math
import numpy as np
import geopandas as gpd
import pandas as pd
warnings.filterwarnings('ignore')
from shapely.geometry import Polygon

# import local modules
sys.path.append("./src")
sys.path.append("../src")
from Fishnet import Fishnet
from ImageDataExporter import ImageDataExporter
# import GeemapUtils as geemap

TILE_SIZE_MILES = 0.25
BATCH_SIZE_MILES = 16  # 16
PERIODS = ["year"]  # ["summer", "year"]
YEARS = [2016] #, 2017, 2018, 2019]

FISHNET_PATH = "/home/ubuntu/air/nonie/capstone/Capstone_JPMorgan/Data/fishnet_filtered.pkl"
FILTER = False
SAVE_DIR = '/home/ubuntu/air/nonie/capstone/Capstone_JPMorgan/Data/Exports'

print("Loading Fishnet... From file: ", FISHNET_PATH, "\n")
fc = Fishnet.load(FISHNET_PATH)

fc = gpd.GeoDataFrame(fc)

dfw = Polygon([(-94.408984375,31.437005897018928), (-94.408984375,33.65934706744288), (-98.177294921875,33.659347067442883), (-98.177294921875,31.437005897018928)])
fc_dfw = fc[fc.intersects(dfw)]

sah = Polygon([(-99.16032395798092,27.66563518068342), (-94.32633958298092,27.66563518068342), (-94.32633958298092,30.922593412992846), \
              (-99.16032395798092,30.922593412992846)])
fc_sah = fc[fc.intersects(sah)]

desert = Polygon([(-102.78581223923092,31.223709273033393), (-99.55583177048092,31.223709273033393), (-99.55583177048092,34.05412455172926), \
              (-102.78581223923092,34.05412455172926)])
fc_desert = fc[fc.intersects(desert)]

concatenated = pd.concat([fc_dfw, fc_sah, fc_desert])
concatenated = concatenated.drop_duplicates()
fc_other = fc[~fc.index.isin(concatenated.index)]

max_rows = 200000

# 3 of them
dfs_dfw = [fc_dfw[i:i+max_rows] for i in range(0, len(fc_dfw), max_rows)]

# 5 of them
dfs_sah = [fc_sah[i:i+max_rows] for i in range(0, len(fc_sah), max_rows)]

# 3 of them
dfs_desert = [fc_desert[i:i+max_rows] for i in range(0, len(fc_desert), max_rows)]

# 12 of them
dfs_other = [fc_other[i:i+max_rows] for i in range(0, len(fc_other), max_rows)]

i = 3
df_which = 'sah' #dfw, sah, desert, other
year_to_run = 2016 #2017, 2018, 2019

if df_which == 'dfw':
    print('Initializing image exporter...', flush=True)
    image_exporter = ImageDataExporter(dfs_dfw[i])
    
    dfs_dfw[i].reset_index(inplace = True)
    dfs_dfw[i]['index_reset'] = dfs_dfw[i].index
    df = pd.DataFrame(dfs_dfw[i])

    # df.to_pickle('/home/ubuntu/air/nonie/capstone/Capstone_JPMorgan/Data/fishnet_filtered_dwf_{}.pkl'.format(i))

elif df_which == 'sah':
    print('Initializing image exporter...', flush=True)
    image_exporter = ImageDataExporter(dfs_sah[i])
    
    dfs_sah[i].reset_index(inplace = True)
    dfs_sah[i]['index_reset'] = dfs_sah[i].index
    df = pd.DataFrame(dfs_sah[i])

    # df.to_pickle('/home/ubuntu/air/nonie/capstone/Capstone_JPMorgan/Data/fishnet_filtered_sah_{}.pkl'.format(i))

elif df_which == 'desert':
    print('Initializing image exporter...', flush=True)
    image_exporter = ImageDataExporter(dfs_desert[i])
    
    dfs_desert[i].reset_index(inplace = True)
    dfs_desert[i]['index_reset'] = dfs_desert[i].index
    df = pd.DataFrame(dfs_desert[i])

    # df.to_pickle('/home/ubuntu/air/nonie/capstone/Capstone_JPMorgan/Data/fishnet_filtered_desert_{}.pkl'.format(i))

elif df_which == 'other':
    print('Initializing image exporter...', flush=True)
    image_exporter = ImageDataExporter(dfs_other[i])
    
    dfs_other[i].reset_index(inplace = True)
    dfs_other[i]['index_reset'] = dfs_other[i].index
    df = pd.DataFrame(dfs_other[i])

    # df.to_pickle('/home/ubuntu/air/nonie/capstone/Capstone_JPMorgan/Data/fishnet_filtered_other_{}.pkl'.format(i))

for pe in PERIODS:
    for year in [year_to_run]:
        if pe == "summer":
            image_exporter.set_date_range(year, year, "05", "10")
        else:
            image_exporter.set_date_range(year, year, "01", "12")
  
        image_exporter.export_images(SAVE_DIR, '{}_{}_{}_{}_export_{}_'.format(df_which, i, pe, year))

print("Done.")