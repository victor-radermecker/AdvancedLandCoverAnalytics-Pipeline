import os
import geopandas as gpd
import pandas as pd
from pathlib import Path
import cv3


class ImageProcessor:
    def __init__(self, fishnet, image_folder, region, batch_ids, img_height, img_width):
        self.fishnet = fishnet
        self.image_folder = image_folder
        self.region = region
        self.batch_ids = batch_ids
        self.img_height = img_height
        self.img_width = img_width
        self.mean_built_df = pd.DataFrame(
            columns=["batch_id", "fishnet_id", "mean_built"]
        )

    def process_fishnet(self):
        for batch_id in self.batch_ids:
            print(f"Processing batch {batch_id}...")
            # Filter the fishnet on the batch
            batch_fishnet = self.fishnet[self.fishnet["batch_id"] == batch_id]

            # Get the image for the batch
            image_path = os.path.join(self.image_folder, f"batch_{batch_id}.tif")
            image = cv2.imread(image_path)

            # Extract the built label (in this example, we assume it's red (255, 0, 0))
            built_label_color = (255, 0, 0)
            built_image = extract_label(image, built_label_color)

            # Get pixel coordinates for the fishnet tiles
            batch_fishnet["ImageCoordinates"] = get_pixel_coordinates(batch_fishnet)

            # Compute the mean pixel value of the built label within each fishnet tile
            mean_built_values = mean_pixel_values(built_image, batch_fishnet)

            # Store the mean "built" of each fishnet tile in a new dataframe
            batch_mean_built_df = pd.DataFrame(
                {
                    "batch_id": batch_id,
                    "fishnet_id": batch_fishnet.index,
                    "mean_built": mean_built_values,
                }
            )
            self.mean_built_df = self.mean_built_df.append(
                batch_mean_built_df, ignore_index=True
            )

        return self.mean_built_df
