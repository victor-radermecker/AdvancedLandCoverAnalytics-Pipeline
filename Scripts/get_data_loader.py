# Data Preparation
import pandas as pd
import sys
import numpy as np
from tqdm import tqdm
import ast

sys.path.append("../src")
from SequenceDataLoader import SequenceDataLoader


# Data Preparation
def get_data_loader(df, IMG_DIR, IMG_SIZE, BATCH_SIZE, N_CHANNELS, tab_data=None):
    labels = [2016, 2017, 2018, 2019, 2020, 2021, 2022]
    filename = "landcover_batchID_"
    list_IDs = [filename + str(s) for s in df["batch_id"].unique()]

    # Iterate over the DataFrame and group tile IDs based on batch IDs
    tile_region_dic = {}
    data = df[df.year == 2017]
    for _, row in tqdm(data.iterrows(), total=len(data)):
        batch_id = int(row["batch_id"])
        tile_id = int(row["tile_id"])
        key = filename + str(batch_id)
        if batch_id not in tile_region_dic:
            tile_region_dic[batch_id] = []
        tile_region_dic[batch_id].append(tile_id)

    # rename the keys of the dictionary to match the list_IDs
    tile_region_dic = {filename + str(k): v for k, v in tile_region_dic.items()}

    # Get the urbanization data for each fishnet
    YEAR = 2022
    TARGET = "urbanization"
    fishnet_urbanization = {}
    data = df[df.year == YEAR][["tile_id", TARGET]]
    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        fishnet_urbanization[int(row["tile_id"])] = row[TARGET]

    # Get the coordinate of each fishnet within each region
    # fishnet_coordinates = fishnet.fishnet["ImageCoordinates"]

    # Get the tiles coordinates
    YEAR = 2017
    TARGET = "ImageCoordinates"
    tile_coordinates = {}
    data = df[df.year == YEAR][["tile_id", TARGET]]
    data[TARGET] = data[TARGET].apply(ast.literal_eval)
    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        tile_coordinates[int(row["tile_id"])] = row[TARGET]

    # Initialize the SequenceDataLoader for training data
    data_loader = SequenceDataLoader(
        labels,
        list_IDs,
        fishnet_urbanization,  # y: target variable
        tile_region_dic,
        tile_coordinates,
        IMG_DIR,
        dim=IMG_SIZE,
        batch_size=BATCH_SIZE,
        n_channels=N_CHANNELS,
        shuffle=True,
        tab_data=tab_data,
    )

    return data_loader
