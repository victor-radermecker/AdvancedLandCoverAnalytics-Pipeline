import os
import geopandas as gpd
import pandas as pd
from typing import List
import cv2
from shapely.geometry import box
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio

tqdm.pandas()


class ImageProcessor:
    def __init__(self, fishnet, image_folder: str, file_name, filtered=False):
        self.filtered = filtered
        self.fh = fishnet
        self.image_folder = image_folder
        self.file_name = file_name

        if self.filtered:
            self.fishnet = self.fh.filtered_fishnet
            self.batch_ids = self.fh.filtered_batches.index
            self.min_batch_id = self.fh.filtered_batches.index.min()
        else:
            self.fishnet = self.fh.fishnet
            self.batch_ids = self.fh.batches.index

    def process_images(self):
        self.fishnet["ImageCoordinates"] = np.nan
        self.fishnet["MeanPixel"] = np.nan

        for batch_id in tqdm(list(self.batch_ids), desc="Processing Images:"):
            image_path = os.path.join(
                self.image_folder, f"{self.file_name}_{batch_id-self.min_batch_id}.tif"
            )
            print(image_path)
            image = imageio.imread(image_path)

            # Extract image dimensions
            self.img_height, self.img_width, _ = image.shape

            # extract the "built" label
            built_label = self.extract_label(
                image, (196, 40, 27)
            )  # Assuming "built" is represented by white pixels red #  [196  40  27] for red in DW

            self.temp_fishnet = self.fishnet[
                self.fishnet["batch_id"] == batch_id
            ].copy()

            self.batch_geometry = self.fh.batches.loc[
                self.fh.batches["batch_id"] == batch_id
            ]["geometry"].bounds.values[0]

            self.temp_fishnet["ImageCoordinates"] = self.get_pixel_coordinates(
                self.temp_fishnet
            )
            self.temp_fishnet["MeanPixel"] = self.get_mean_pixel_value(
                built_label, self.temp_fishnet
            )
            self.fishnet.update(self.temp_fishnet)

    def get_pixel_coordinates(self, df):
        # Use the apply() method with axis=1 to apply the latlong_to_pixel function to each row
        image_coordinates = df.apply(
            lambda row: self.latlong_to_pixel(
                self.batch_geometry, row["geometry"].bounds
            ),
            axis=1,
        )
        return image_coordinates

    def latlong_to_pixel(self, batch_coords, tile_coords):
        min_lon, min_lat, max_lon, max_lat = batch_coords
        xmin, ymin, xmax, ymax = tile_coords

        # Normalize the bounding box coordinates
        x_min_pixel = int((xmin - min_lon) / (max_lon - min_lon) * self.img_width)
        x_max_pixel = int((xmax - min_lon) / (max_lon - min_lon) * self.img_width)
        y_min_pixel = int(
            (1 - (ymax - min_lat) / (max_lat - min_lat)) * self.img_height
        )
        y_max_pixel = int(
            (1 - (ymin - min_lat) / (max_lat - min_lat)) * self.img_height
        )

        return x_min_pixel, y_min_pixel, x_max_pixel, y_max_pixel

    def get_mean_pixel_value(self, matrix, df):
        # Use the apply() method with axis=1 to apply the latlong_to_pixel function to each row
        mean_pixel = df.apply(
            lambda row: self.mean_pixel_value(matrix, row["ImageCoordinates"]),
            axis=1,
        )
        return mean_pixel

    def mean_pixel_value(self, matrix: np.ndarray, bounds: list):
        xmin, ymin, xmax, ymax = bounds
        submatrix = matrix[ymin:ymax, xmin:xmax]
        mean_value = np.mean(submatrix)
        return mean_value

    def extract_label(self, image, color):
        red_pixels = np.all(
            [
                image[:, :, 0] == color[0],  # Red channel
                image[:, :, 1] == color[1],  # Green channel
                image[:, :, 2] == color[2],  # Blue channel
            ],
            axis=0,
        )
        extracted_image = np.zeros_like(image)
        extracted_image[red_pixels] = [
            255,
            255,
            255,
        ]  # Extracted image is black & white
        return extracted_image

    def get_unique_colors(self, image):
        # Reshape the image array into a 2D array of shape (num_pixels, 3)
        reshaped_image = image.reshape(-1, image.shape[-1])

        # Find unique color triples using numpy
        unique_colors = np.unique(reshaped_image, axis=0)

        return unique_colors
