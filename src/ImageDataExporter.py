import ee
from tqdm import tqdm
import numpy as np
import pandas as pd
import GeemapUtils as geemap
import random
import math
import os
from googleapiclient.errors import HttpError

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

        if startYear == 2016:
            lastYearStartDate = f"{startYear-1}-07-01"
            lastYearEndDate = f"{endYear-1}-{endMonth}-01"

            twoYearsAgoStartDate = None
            twoYearsAgoEndDate = None            
        else:
            lastYearStartDate = f"{startYear-1}-{startMonth}-01"
            lastYearEndDate = f"{endYear-1}-{endMonth}-01"

            if startYear == 2017:
                twoYearsAgoStartDate = f"{startYear-2}-07-01"
                twoYearsAgoEndDate = f"{endYear-2}-{endMonth}-01"

            else:
                twoYearsAgoStartDate = f"{startYear-2}-{startMonth}-01"
                twoYearsAgoEndDate = f"{endYear-2}-{endMonth}-01"

        self.startDate = startDate
        self.endDate = endDate

        self.lastYearStartDate = lastYearStartDate
        self.lastYearEndDate = lastYearEndDate
        
        self.twoYearsAgoStartDate = twoYearsAgoStartDate
        self.twoYearsAgoEndDate = twoYearsAgoEndDate

    def export_images(self, save_dir, fname_prefix, checkpoint_file = None, fname = 'raw_value_counts.csv'):

        if checkpoint_file:
            self.df = pd.read_csv(os.path.join(save_dir, checkpoint_file), index_col=0)
            data = self.df.to_dict()

            filtered_df = self.df[self.df['expectation_built'] == -1]

            if not filtered_df.empty:
                starting_index = filtered_df.index[0]
            else:
                print('Checkpoint file already finished. Exiting.')
                return

        else:
            data = {'tile_no': [i for i in range(self.fh.shape[0])], 'expectation_built': [-1 for _ in range(self.fh.shape[0])], \
                    'perc_built': [-1 for _ in range(self.fh.shape[0])], 'ntl': [-1 for _ in range(self.fh.shape[0])], \
                    'perc_new_urban_1year': [-1 for _ in range(self.fh.shape[0])], 'perc_new_urban_2year': [-1 for _ in range(self.fh.shape[0])]}
            starting_index = 0
        
        dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filter(ee.Filter.date(self.startDate, self.endDate))   
        dw_last_year = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filter(ee.Filter.date(self.lastYearStartDate, self.lastYearEndDate))   
        if self.twoYearsAgoStartDate:
            dw_2_years_ago = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filter(ee.Filter.date(self.twoYearsAgoStartDate, self.twoYearsAgoEndDate)) 
        else:
            dw_2_years_ago = None
        viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate(self.startDate, self.endDate).select('avg_rad')

        for i in tqdm(range(starting_index, self.fh.shape[0])):

            try:
                region =  ee.Geometry.Rectangle(self.fh['geometry'].iloc[i].bounds)

                dw_clip = dw.filterBounds(region)

                ### Expectation Built
                reducer = ee.Reducer.mean()
                built = dw_clip.select('built')
                mean_built = built.reduce(reducer)

                expected_built = mean_built.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region,
                    scale=10
                ).getInfo()['built_mean']

                data['expectation_built'][i] = expected_built

                ### New Urban
                try:
                    dw_clip_last_year = dw_last_year.filterBounds(region)
                    built_last_year = dw_clip_last_year.select('built')
                    mean_built_last_year = built_last_year.reduce(reducer)
                    newUrban1year = mean_built_last_year.lt(0.15).And(mean_built.gt(0.35))

                    newUrban1year = newUrban1year.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=region,
                        scale=10
                    ).getInfo()['built_mean']

                    if newUrban1year:
                        data['perc_new_urban_1year'][i] = newUrban1year * 100
                except:
                    data['perc_new_urban_1year'][i] = -1000

                try:
                    if dw_2_years_ago:
                        dw_clip_2_years_ago = dw_2_years_ago.filterBounds(region)
                        built_2_years_ago = dw_clip_2_years_ago.select('built')
                        mean_built_2_years_ago = built_2_years_ago.reduce(reducer)
                        newUrban2year = mean_built_2_years_ago.lt(0.15).And(mean_built.gt(0.35))

                        newUrban2year = newUrban2year.reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=region,
                            scale=10
                        ).getInfo()['built_mean']

                        if newUrban2year:
                            data['perc_new_urban_2year'][i] = newUrban2year * 100
                except:
                    data['perc_new_urban_2year'][i] = -1000

                ### Proportion Built
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

                ### NTL
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