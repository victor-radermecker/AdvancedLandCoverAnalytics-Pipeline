import os
import geopandas as gpd
import pandas as pd
from typing import List
import cv2
from shapely.geometry import box
from PIL import Image
import numpy as np
from tqdm import tqdm

# The provided functions remain unchanged
# latlong_to_pixel, get_pixel_coordinates, mean_pixel_values, get_unique_colors, extract_label


class ImageProcessor:
    def __init__(
        self, fishnet: gpd.GeoDataFrame, image_folder: str, batch_ids: List[int]
    ):
        assert isinstance(fishnet, gpd.GeoDataFrame), "fishnet must be a GeoDataFrame."
        self.fishnet = fishnet
        self.image_folder = image_folder
        self.batch_ids = batch_ids

    def process_images(self):
        mean_built_list = []

        for batch_id in self.batch_ids:
            image_path = os.path.join(self.image_folder, f"batch_{batch_id}.tif")
            image = cv2.imread(image_path)

            # Extract image dimensions
            img_height, img_width, _ = image.shape

            built_label = self.extract_label(
                image, (255, 255, 255)
            )  # Assuming "built" is represented by white pixels red #  [196  40  27] for red in DW
            pixel_coordinates = self.get_pixel_coordinates(self.fishnet)

            self.fishnet["ImageCoordinates"] = pixel_coordinates
            mean_pixel_values_list = self.mean_pixel_values(built_label, self.fishnet)

            for fishnet_id, mean_built in zip(
                self.fishnet.index, mean_pixel_values_list
            ):
                mean_built_list.append(
                    {
                        "batch_id": batch_id,
                        "fishnet_id": fishnet_id,
                        "mean_built": mean_built,
                    }
                )

        mean_built_df = pd.DataFrame(mean_built_list)
        return mean_built_df

    def get_pixel_coordinates(self, geodataframe):
        region = self.fishnet.batches.geometry.bounds
        pixel_coordinates = geodataframe["geometry"].apply(
            lambda x: latlong_to_pixel(
                x.bounds, region, (self.img_height, self.img_width)
            )
        )
        return pixel_coordinates

    def mean_pixel_values(self, img_arr, bounding_boxes_gdf):
        img = Image.fromarray(cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB))
        bounding_boxes_gdf["PixelCoordinates"] = bounding_boxes_gdf[
            "ImageCoordinates"
        ].apply(lambda x: (int(x[0]), int(x[1]), int(x[2]), int(x[3])))
        mean_pixel_values = []
        with tqdm(total=len(bounding_boxes_gdf)) as pbar:
            for idx, row in bounding_boxes_gdf.iterrows():
                pixel_box = img.crop(row["PixelCoordinates"])
                mean_pixel_value = np.mean(pixel_box)
                mean_pixel_values.append(mean_pixel_value)
                pbar.update(1)
        return mean_pixel_values

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


# @ TODO !!
def latlong_to_pixel(bbox, region, img_shape):
    """
    Convert lat/long coordinates to pixel coordinates.

    Parameters:
    bbox (tuple): A tuple of (xmin, ymin, xmax, ymax) representing the bounding box.
    region (list): A list of [x1, x2, x3, x4] representing the lat/long coordinates of the 4 image corners.
    img_shape (tuple): A tuple of (height, width) representing the image shape.

    Returns:
    tuple: A tuple of (xmin, ymin, xmax, ymax) representing the pixel coordinates of the bounding box.
    """
    img_height, img_width = img_shape
    x1, x2, x3, x4 = region

    min_lon, max_lon = min(x1, x3), max(x1, x3)
    min_lat, max_lat = min(x2, x4), max(x2, x4)
    xmin, ymin, xmax, ymax = bbox

    # Normalize the bounding box coordinates
    x_min_pixel = int((xmin - min_lon) / (max_lon - min_lon) * img_width)
    x_max_pixel = int((xmax - min_lon) / (max_lon - min_lon) * img_width)
    y_min_pixel = int((1 - (ymax - min_lat) / (max_lat - min_lat)) * img_height)
    y_max_pixel = int((1 - (ymin - min_lat) / (max_lat - min_lat)) * img_height)

    return x_min_pixel, y_min_pixel, x_max_pixel, y_max_pixel
