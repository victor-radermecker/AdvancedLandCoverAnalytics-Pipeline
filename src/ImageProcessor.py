import os
import numpy as np
from tqdm.auto import tqdm
import imageio

tqdm.pandas()


class ImageProcessor:
    def __init__(
        self,
        fishnet,
        year,
        image_folder: str,
        file_name: str,
        feature_name: str,
        filtered=False,
    ):
        self.filtered = filtered
        self.year = year
        self.fh = fishnet
        self.image_folder = image_folder
        self.file_name = file_name
        self.feature_name = feature_name

        if self.filtered:
            self.fishnet = self.fh.filtered_fishnet
            self.batch_ids = self.fh.filtered_batches.index
        else:
            self.fishnet = self.fh.fishnet
            self.batch_ids = self.fh.batches.index

    def process_images(self):
        self.fishnet["ImageCoordinates"] = np.nan
        self.fishnet[self.feature_name] = np.nan

        for batch_id in tqdm(list(self.batch_ids), desc="Processing Images"):
            image_path = os.path.join(
                self.image_folder, f"{self.file_name}_{batch_id}.tif"
            )
            image = imageio.imread(image_path)

            # Extract image dimensions
            self.img_height, self.img_width, _ = image.shape

            # extract the "built" label
            built_label = self.extract_label(
                image, (196, 40, 27)
            )  # Assuming "built" is represented by white pixels red #  [196  40  27] for red in DW

            temp_fishnet = self.fishnet[
                self.fishnet["batch_id"] == batch_id
            ].copy()

            self.batch_geometry = self.fh.batches.loc[
                self.fh.batches["batch_id"] == batch_id
            ]["geometry"].bounds.values[0]

            temp_fishnet["ImageCoordinates"] = self.get_pixel_coordinates(
                temp_fishnet
            )
            temp_fishnet[self.feature_name] = self.get_mean_pixel_value(
                built_label, temp_fishnet
            )
            self.fishnet.update(temp_fishnet)

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
    
    def cnn_partition_images(self, warning=True, show_progress=True):

        # Write a message to the user: "Warning, this code will create a new folder CNN in the image folder, which may take a lot of space on the hard drive. Continue?"
        # If the user says yes, continue, otherwise, stop the code
        if warning:
            answer = input("Warning, this code will create a new folder CNN in the image folder, which may take a lot of space on the hard drive. Continue? Yes or No?")
        else:
            answer = "Yes"

        if answer == "Yes":
            progress_bar = tqdm(list(self.batch_ids), ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}', disable=show_progress)

            for batch_id in progress_bar:
                image_path = os.path.join(
                    self.image_folder, f"{self.file_name}_{batch_id}.tif"
                )
                image = imageio.imread(image_path)

                # extract all fishnet tile ids in the current batch
                fishnet_tiles = self.fishnet[
                    self.fishnet["batch_id"] == batch_id
                ]["id"]

                for tile_id in fishnet_tiles:
                    tile = self.fishnet[self.fishnet["id"] == tile_id]
                    xmin, ymin, xmax, ymax = tile["ImageCoordinates"].values[0]
                    subimage = image[ymin:ymax, xmin:xmax]

                    # check if the directory self.image_folder + 'CNN' exists, otherwise, create it
                    if not os.path.exists(self.image_folder + f'../../CNN/{self.year}/'):
                        os.makedirs(self.image_folder + f'../../CNN/{self.year}/')
                        print("Directory " , self.image_folder + f'../CNN/{self.year}/' ,  " Created ")

                    export_path = self.image_folder + f'../../CNN/{self.year}/' + f"/{int(tile_id)}.tif"
                    imageio.imwrite(export_path, subimage)

        else:
            print("Aborted.")

