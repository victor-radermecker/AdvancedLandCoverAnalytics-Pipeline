import ee
from tqdm import tqdm
import numpy as np
import pandas as pd
import GeemapUtils as geemap
import random
import math
import os
from googleapiclient.errors import HttpError

PROB = True
DW = False

ee.Initialize()

class ImageDataExporter:
    def __init__(self, fishnet, scale=10):
        self.fh = fishnet
        self.scale = scale
        self.fileFormat = "GeoTIFF"
        self.maxPixels = 1e10
        
    def set_date_range(self, startYear, endYear, startMonth, endMonth):
        startDate = f"{startYear}-{startMonth}-01"
        endDate = f"{endYear}-{endMonth}-01"

        self.startDate = startDate
        self.endDate = endDate

    def export_images(self, save_dir, fname_prefix, fname = 'raw_value_counts.csv'):
        
        if DW:
            if PROB:
                data = {'tile_no': [i for i in range(self.fh.shape[0])], 'expectation_built': [-1 for _ in range(self.fh.shape[0])]}
            else:
                data = {'tile_no': [i for i in range(self.fh.shape[0])], 'perc_built': [-1 for _ in range(self.fh.shape[0])]}
        else:
            data = {'tile_no': [i for i in range(self.fh.shape[0])], 'ntl': [-1 for _ in range(self.fh.shape[0])]}
        # self.df = pd.DataFrame(data)

        if DW:
            dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filter(
                    ee.Filter.date(self.startDate, self.endDate)
            )   
        else:
            viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate(self.startDate, self.endDate).select('avg_rad')

        for i in tqdm(range(self.fh.shape[0])):

            try:
                region =  ee.Geometry.Rectangle(self.fh['geometry'].iloc[i].bounds)

                if DW:
                    dw_clip = dw.filterBounds(region)

                    if PROB:
                        reducer = ee.Reducer.mean()

                        built = dw_clip.select('built')
                        mean_built = built.reduce(reducer)

                        expected_built = mean_built.reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=region,
                            scale=10
                        ).getInfo()['built_mean']

                        data['expectation_built'][i] = expected_built

                    else:
                        reducer = ee.Reducer.mode()

                        classification = dw_clip.select('label')
                        dwComposite = classification.reduce(reducer)

                        builtArea = dwComposite.eq(6)

                        totalPixels = builtArea.reduceRegion(
                            reducer=ee.Reducer.count(),
                            geometry=region,
                            scale=10
                        ).getInfo()['label_mode']

                        builtAreaMasked = builtArea.selfMask()
                        builtAreaPixels = builtAreaMasked.reduceRegion(
                            reducer=ee.Reducer.count(),
                            geometry=region,
                            scale=10
                        ).getInfo()['label_mode']

                        if totalPixels != 0:
                            percent = builtAreaPixels / totalPixels * 100

                        data['perc_built'][i] = percent
                else:
                    viirs_clip = viirs.filterBounds(region)
                    reducer = ee.Reducer.mean()
                    viirs_mean = viirs_clip.reduce(reducer)

                    ntl = viirs_mean.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=region,
                        scale=400
                    ).getInfo()['avg_rad_mean']

                    data['ntl'][i] = ntl

            except HttpError as e:
                if e.resp.status == 400 and "Computation timed out" in e._get_reason():
                    print("Computation timed out. Proceeding to the next step.")
                else:
                    print("An HttpError occurred, but it's not the one we expected.")
 
            if i % 10000 == 0:
                self.df = pd.DataFrame(data)
                self.df.to_csv(os.path.join(save_dir, 'sub_' + fname_prefix + fname))

        self.df = pd.DataFrame(data)
        self.df.to_csv(os.path.join(save_dir, fname_prefix + fname))