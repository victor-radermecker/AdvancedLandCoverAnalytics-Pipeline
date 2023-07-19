import os
import numpy as np
from tqdm.auto import tqdm
import imageio
from scipy.stats import entropy

tqdm.pandas()


class ImageProcessor:
    def __init__(
        self,
        fishnet,
        filtered=False,
    ):
        self.filtered = filtered
        self.fh = fishnet

        if self.filtered:
            self.fishnet = self.fh.filtered_fishnet
            self.batch_ids = self.fh.filtered_batches.index
        else:
            self.fishnet = self.fh.fishnet
            self.batch_ids = self.fh.batches.index

    def assign_fishnet_tiles_to_pixels(self, image_folder, file_name):
        self.fishnet["ImageCoordinates"] = np.nan
        for batch_id in tqdm(list(self.batch_ids), desc="Processing Images"):
            image_path = os.path.join(image_folder, f"{file_name}_{batch_id}.tif")
            image = imageio.imread(image_path)
            if image is None:
                raise Exception("Error reading image.")

            # Extract image dimensions
            self.img_height, self.img_width, _ = image.shape

            temp_fishnet = self.fishnet[self.fishnet["batch_id"] == batch_id].copy()

            self.batch_geometry = self.fh.batches.loc[
                self.fh.batches["batch_id"] == batch_id
            ]["geometry"].bounds.values[0]

            temp_fishnet["ImageCoordinates"] = self.get_pixel_coordinates(temp_fishnet)

            self.fishnet.update(temp_fishnet)

        self.fishnet["Width"] = self.fishnet["ImageCoordinates"].apply(
            lambda x: x[2] - x[0]
        )
        self.fishnet["Height"] = self.fishnet["ImageCoordinates"].apply(
            lambda x: x[3] - x[1]
        )

    def compute_mean_tile_entropy_urbanization(
        self, image_folder, file_name, feature1_name, feature2_name
    ):
        self.fishnet[feature1_name] = np.nan
        self.fishnet[feature2_name] = np.nan

        for batch_id in tqdm(list(self.batch_ids), desc="Processing Images"):
            image_path = os.path.join(image_folder, f"{file_name}_{batch_id}.tif")
            image = imageio.imread(image_path)
            if image is None:
                raise Exception("Error reading image.")

            # Extract image dimensions
            self.img_height, self.img_width, _ = image.shape

            # extract the "built" label
            built_label = self.extract_label(
                image, (196, 40, 27)
            )  # Assuming "built" is represented by white pixels red #  [196  40  27] for red in DW

            temp_fishnet = self.fishnet[self.fishnet["batch_id"] == batch_id].copy()

            self.batch_geometry = self.fh.batches.loc[
                self.fh.batches["batch_id"] == batch_id
            ]["geometry"].bounds.values[0]

            (
                temp_fishnet[feature1_name],
                temp_fishnet[feature2_name],
            ) = self.get_mean_pixel_entropy_values(built_label, temp_fishnet)

            self.fishnet.update(temp_fishnet)

    def get_pixel_coordinates(self, df):
        # Use the apply() method with axis=1 to apply the latlong_to_pixel function to each row
        image_coordinates = df.apply(
            lambda row: self.latlong_to_pixel(
                self.batch_geometry, row["geometry"].bounds, row["id"]
            ),
            axis=1,
        )

        # raise Error if image coordinates is nan
        if image_coordinates.isna().any():
            raise Exception("Error: Image coordinates are nan.")

        return image_coordinates

    def latlong_to_pixel(self, batch_coords, tile_coords, id):
        min_lon, min_lat, max_lon, max_lat = batch_coords  # long/lat format
        xmin, ymin, xmax, ymax = tile_coords  # long/lat format

        # check if xmin > min_lon, xmax>xmin, xmax < max_lon, ymin > min_lat, ymax > ymin, ymax < max_lat
        if (
            xmin < min_lon
            #            or xmax < xmin
            or xmax > max_lon
            or ymin < min_lat
            #            or ymax < ymin
            or ymax > max_lat
        ):
            print("Tile: ", id)
            print("Xmin: ", xmin)
            print("Xmax: ", xmax)
            print("Ymin: ", ymin)
            print("Ymax: ", ymax)
            print("Min Lon: ", min_lon)
            print("Max Lon: ", max_lon)
            print("Min Lat: ", min_lat)
            print("Max Lat: ", max_lat)
            raise Exception("Error: Tile coordinates are not within batch coordinates.")

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

    def get_mean_pixel_entropy_values(self, matrix, df):
        # Use the apply() method with axis=1 to apply the function to each row
        mean_pixel = df.apply(
            lambda row: self.mean_pixel_value(matrix, row["ImageCoordinates"]),
            axis=1,
        )
        entropy = df.apply(
            lambda row: self.entropy(matrix, row["ImageCoordinates"]), axis=1
        )
        return mean_pixel, entropy

    def mean_pixel_value(self, matrix: np.ndarray, bounds: list):
        xmin, ymin, xmax, ymax = bounds
        submatrix = matrix[ymin:ymax, xmin:xmax]
        mean_value = np.mean(submatrix)
        return mean_value

    def entropy(self, matrix: np.ndarray, bounds: list):
        xmin, ymin, xmax, ymax = bounds
        submatrix = matrix[ymin:ymax, xmin:xmax]
        flat_submatrix = submatrix.ravel()
        probabilities = np.bincount(flat_submatrix) / len(flat_submatrix)
        entropy_value = entropy(probabilities, base=2)
        return entropy_value

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

    def generate_processed_csv(self, save_path, file_name):
        # Compute Urbanization Rate
        years = list(range(2016, 2023))
        for yr in years[1:]:
            self.fishnet.compute_difference(
                f"MeanPixel_{yr}", f"MeanPixel_{yr-1}", filtered=True, normalize=True
            )

        # rename MeanPixel_2017-MeanPixel_2016 to urbanization_rate_2016
        for yr in years[1:]:
            self.fishnet.filtered_fishnet.rename(
                columns={f"MeanPixel_{yr}-MeanPixel_{yr-1}": f"urbanization_rate_{yr}"},
                inplace=True,
            )

        for yr in years:
            self.fishnet.filtered_fishnet.rename(
                columns={f"MeanPixel_{yr}": f"urbanization_{yr}"}, inplace=True
            )

        self.fishnet.filtered_fishnet.rename(columns={"id": "tile_id"}, inplace=True)
        self.fishnet.filtered_fishnet["tile_id"] = self.fishnet.filtered_fishnet[
            "tile_id"
        ].astype(int)
        self.fishnet.filtered_fishnet["batch_id"] = self.fishnet.filtered_fishnet[
            "batch_id"
        ].astype(int)

        # Extracting Lat Long coordinates
        self.fishnet.filtered_fishnet["centroid"] = self.fishnet.filtered_fishnet[
            "geometry"
        ].apply(lambda x: x.centroid)
        self.fishnet.filtered_fishnet["Lat"] = self.fishnet.filtered_fishnet[
            "centroid"
        ].apply(lambda x: x.y)
        self.fishnet.filtered_fishnet["Lon"] = self.fishnet.filtered_fishnet[
            "centroid"
        ].apply(lambda x: x.x)

        vars1 = ["tile_id", "batch_id"] + [
            f"urbanization_rate_{year}" for year in range(2017, 2023)
        ]
        vars2 = ["tile_id", "batch_id"] + [
            f"urbanization_{year}" for year in range(2016, 2023)
        ]

        data = self.fishnet.filtered_fishnet[vars1].melt(
            id_vars=["tile_id", "batch_id"],
            var_name="year",
            value_name="urbanization_rate",
        )
        data["year"] = data["year"].str[-4:]
        data["urbanization"] = (
            self.fishnet.filtered_fishnet[vars2].melt(
                id_vars=["tile_id", "batch_id"],
                var_name="year",
                value_name="urbanization",
            )["urbanization"]
            / 255
        )

        # data['Lat'] is the latitude of the centroid of the tile in fc.filtered_fishnet['Lat'] joint
        data = data.merge(
            self.fishnet.filtered_fishnet[["tile_id", "batch_id", "Lat", "Lon"]],
            on=["tile_id", "batch_id"],
        )

        # Save result
        data.to_csv(save_path + file_name + ".csv", index=False)

        # Save Metadata
        with open(save_path + file_name + ".txt", "w") as f:
            f.write("\n\nGeneral Fishnet Information:\n")
            f.write(str(self.fishnet.fishnet_info(return_=True)) + "\n\n")
            f.write("\n\nGeneral Batch Information:\n")
            f.write(str(self.fishnet.batch_info(return_=True)))
            if self.fishnet.filtered:
                f.write("\n\n\nGeneral Filter Information:")
                f.write("\nFiltered region: " + str(self.fishnet.filter_region))
                f.write("\nNumber of rows: " + str(self.fishnet.filtered_fishnet_rows))
                f.write("\nNumber of cols: " + str(self.fishnet.filtered_fishnet_cols))

    def cnn_partition_images(
        self,
        image_folder,
        file_name,
        year,
        img_size,
        warning=True,
        show_progress=True,
    ):
        # Write a message to the user: "Warning, this code will create a new folder CNN in the /Image/ folder, which may take a lot of space on the hard drive. Continue?"
        # If the user says yes, continue, otherwise, stop the code
        if warning:
            answer = input(
                "Warning, this code will create a new folder CNN in the /image_folder/ folder, which may require consequent storage on the hard drive. Continue? Yes or No?"
            )
        else:
            answer = "Yes"

        if answer == "Yes":
            progress_bar = tqdm(
                list(self.batch_ids),
                ncols=80,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
                disable=show_progress,
            )

            for batch_id in progress_bar:
                image_path = os.path.join(
                    image_folder,
                    year,
                    "Final",
                    f"{file_name}_{batch_id}.tif",
                )
                image = imageio.imread(image_path)

                # extract all fishnet tile ids in the current batch
                fishnet_tiles = self.fishnet[self.fishnet["batch_id"] == batch_id]["id"]

                for tile_id in fishnet_tiles:
                    tile = self.fishnet[self.fishnet["id"] == tile_id]
                    xmin, ymin, xmax, ymax = tile["ImageCoordinates"].values[0]
                    subimage = image[ymin:ymax, xmin:xmax]

                    if subimage.shape[0] < 30 or subimage.shape[1] < 30:
                        raise Exception(
                            f"Subimage {tile_id} in batch {batch_id} is too small. Please check the image and fishnet."
                        )

                    # Crop image to img_size
                    subimage = subimage[: img_size[0], : img_size[1]]

                    # check if the directory self.image_folder + 'CNN' exists, otherwise, create it
                    if not os.path.exists(image_folder + f"./CNN/{year}/"):
                        os.makedirs(image_folder + f"./CNN/{year}/")
                        print(
                            "Directory ",
                            image_folder + f"./CNN/{year}/",
                            " Created ",
                        )

                    export_path = (
                        image_folder + f"./CNN/{year}/" + f"/{int(tile_id)}.tif"
                    )
                    imageio.imwrite(export_path, subimage)

        else:
            print("Aborted.")
