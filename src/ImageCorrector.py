import os
from PIL import Image
import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm
import sys
from matplotlib import pyplot as plt


class ImageCorrector:
    def __init__(self, base_path, verbose=True):
        self.base_path = base_path
        self.years = sorted(self.list_folders())
        self.verbose = verbose
        if self.verbose:
            self.summary()

    def summary(self):
        # create a table
        table = PrettyTable()
        table.field_names = [
            "Year",
            "Summer Images",
            "Year Images",
            "Summer Black Pixel %",
            "Year Black Pixel %",
            "Summer Snow Pixel %",
            "Year Snow Pixel %",
        ]

        # populate the table
        for year in tqdm(self.years, desc="Loading images...", file=sys.stdout):
            summer_path = os.path.join(self.base_path, year, "Summer")
            year_path = os.path.join(self.base_path, year, "Year")

            summer_black_pixel_proportion = self.compute_color_pixels_proportion(
                summer_path, color=[0, 0, 0]
            )
            year_black_pixel_proportion = self.compute_color_pixels_proportion(
                year_path, color=[0, 0, 0]
            )

            summer_snow_pixel_proportion = self.compute_color_pixels_proportion(
                summer_path, color=[179, 159, 225]
            )
            year_snow_pixel_proportion = self.compute_color_pixels_proportion(
                year_path, color=[179, 159, 225]
            )

            table.add_row(
                [
                    year,
                    self.count_images(summer_path),
                    self.count_images(year_path),
                    "{:.4%}".format(summer_black_pixel_proportion),
                    "{:.4%}".format(year_black_pixel_proportion),
                    "{:.4%}".format(summer_snow_pixel_proportion),
                    "{:.4%}".format(year_snow_pixel_proportion),
                ]
            )

        # print the table
        table.sortby = "Year"
        print(table)

    def summary_final(self):
        # create a table
        table = PrettyTable()
        table.field_names = [
            "Year",
            "Final Images",
            "Final Black Pixel %",
            "Final Snow Pixel %",
        ]

        # populate the table
        for year in tqdm(self.years, desc="Loading images...", file=sys.stdout):
            final_path = os.path.join(self.base_path, year, "Final")
            final_black_pixel_proportion = self.compute_color_pixels_proportion(
                final_path, color=[0, 0, 0]
            )

            final_snow_pixel_proportion = self.compute_color_pixels_proportion(
                final_path, color=[179, 159, 225]
            )

            table.add_row(
                [
                    year,
                    self.count_images(final_path),
                    "{:.4%}".format(final_black_pixel_proportion),
                    "{:.4%}".format(final_snow_pixel_proportion),
                ]
            )

        # print the table
        table.sortby = "Year"
        print(table)

    def list_folders(self):
        return [
            entry
            for entry in os.listdir(self.base_path)
            if os.path.isdir(os.path.join(self.base_path, entry))
            and entry.startswith("20")
        ]

    def count_images(self, path):
        images = [
            file
            for file in os.listdir(path)
            if file.lower().endswith((".tif", ".png", ".jpg", ".jpeg"))
        ]
        return len(images)

    def list_files(self, path):
        return [
            entry
            for entry in os.listdir(path)
            if os.path.isfile(os.path.join(path, entry)) and entry.endswith(".tif")
        ]

    def read_image(self, path):
        img = Image.open(path)
        data = np.asarray(img, dtype="int32")
        return data

    def save_image(self, path, img):
        # save the image with PIL
        im = Image.fromarray(img.astype("uint8"))
        im.save(path)

    def correct_images(self):
        print("Imputing Summer missing pixels with Year data. Processing images...")
        for year in self.years:  # years is a string"
            self.set_final_path(year)
            # Input paths
            summer_path = os.path.join(self.base_path, str(year), "Summer")
            year_path = os.path.join(self.base_path, str(year), "Year")

            # Output paths
            if not os.path.exists(os.path.join(self.base_path, str(year), "Final")):
                os.makedirs(os.path.join(self.base_path, str(year), "Final"))
            output_path = os.path.join(self.base_path, str(year), "Final")

            summer_files = sorted(self.list_files(summer_path))
            year_files = sorted(self.list_files(year_path))

            for summer_file, year_file in tqdm(
                zip(summer_files, year_files),
                desc=f"Correcting images for year {year}",
                file=sys.stdout,
                total=len(summer_files),
            ):
                summer_img = self.read_image(os.path.join(summer_path, summer_file))
                year_img = self.read_image(os.path.join(year_path, year_file))

                if summer_img is None or year_img is None:
                    raise Exception("Error reading image.")

                if summer_img.shape != year_img.shape:
                    raise Exception("Images have different dimensions.")

                result_img = np.where(summer_img != [0, 0, 0], summer_img, year_img)

                if int(year) > 2016:
                    previous_img = self.read_image(
                        os.path.join(
                            self.base_path, str(int(year) - 1), "Final", summer_file
                        )
                    )
                    result_img = self.discard_deurbanization(previous_img, result_img)

                # Save locally
                self.save_image(os.path.join(output_path, summer_file), result_img)

        if self.verbose:
            self.summary_final()

    def set_final_path(self, year):
        final_path = os.path.join(self.base_path, year, "Final")
        # Creating Final Directory if it doesn't exist
        if not os.path.exists(final_path):
            print(f"Creating directory {final_path}.")
            os.makedirs(final_path)

    def discard_deurbanization(self, previous_img, result_img):
        # avoid deurbanization in the data
        return np.where(previous_img != [196, 40, 27], result_img, previous_img)

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

    # @TO_OPTIMIZE
    def compute_color_pixels_proportion(self, path, color=[0, 0, 0]):
        total_pixels = 0
        color_pixels = 0

        files = self.list_files(path)

        for file in files:
            img = self.read_image(os.path.join(path, file))

            if img is None:
                print(file)
                raise Exception("Error reading image.")

            total_pixels += np.size(img)
            color_pixels += np.sum(img == color)

        proportion = color_pixels / total_pixels if total_pixels > 0 else 0
        return proportion
