import os
import cv2
import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm
import sys
from matplotlib import pyplot as plt


class ImageCorrector:
    def __init__(self, base_path):
        self.base_path = base_path
        self.years = self.list_folders()
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
            and entry.startswith("export_")
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
        return cv2.imread(path, cv2.IMREAD_COLOR)

    def save_image(self, path, img):
        cv2.imwrite(path, img)

    def correct_images(self):
        print("Imputing Summer missing pixels with Year data. Processing images...")
        for year in self.years:
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
                desc=f"Processing images for year {year}",
                file=sys.stdout,
                total=len(summer_files),
            ):
                summer_img = self.read_image(os.path.join(summer_path, summer_file))
                year_img = self.read_image(os.path.join(year_path, year_file))

                if summer_img is None or year_img is None:
                    raise Exception("Error reading image.")

                if summer_img.shape != year_img.shape:
                    raise Exception("Images have different dimensions.")

                result_img = np.where(summer_img != 0, summer_img, year_img)

                # Save locally
                self.save_image(os.path.join(output_path, summer_file), result_img)

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


# import os
# import dropbox
# import cv2
# import numpy as np


# class ImageCorrector:
#     def __init__(self, dbx_token_path):
#         # Read API Access Token from file
#         with open(dbx_token_path) as f:
#             dbx_token = f.read()

#         self.dbx = dropbox.Dropbox(dbx_token)

#     def list_files(self, path):
#         try:
#             res = self.dbx.files_list_folder(path).entries
#             return [
#                 entry.name
#                 for entry in res
#                 if isinstance(entry, dropbox.files.FileMetadata)
#             ]
#         except dropbox.exceptions.ApiError as err:
#             print(f"Dropbox API Error: {err}")
#             return []

#     def read_image(self, path):
#         try:
#             metadata, res = self.dbx.files_download(path)
#             img_arr = np.asarray(bytearray(res.content), dtype=np.uint8)
#             img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
#             return img
#         except dropbox.exceptions.ApiError as err:
#             print(f"Dropbox API Error: {err}")
#             return None

#     def upload_image(self, path, file_name):
#         try:
#             with open(file_name, "rb") as f:
#                 self.dbx.files_upload(
#                     f.read(), path, mode=dropbox.files.WriteMode("overwrite")
#                 )
#         except dropbox.exceptions.ApiError as err:
#             print(f"Dropbox API Error: {err}")

#     def process_images(self, base_path, years):
#         for year in years:
#             summer_path = os.path.join(base_path, str(year), "Summer")
#             yearly_path = os.path.join(base_path, str(year), "Yearly")
#             output_path = os.path.join(base_path, str(year), "Final")

#             summer_files = self.list_files(summer_path)
#             yearly_files = self.list_files(yearly_path)

#             for summer_file, yearly_file in zip(summer_files, yearly_files):
#                 summer_img = self.read_image(os.path.join(summer_path, summer_file))
#                 yearly_img = self.read_image(os.path.join(yearly_path, yearly_file))

#                 if summer_img is None or yearly_img is None:
#                     raise Exception("Error reading image.")

#                 result_img = np.where(summer_img == 0, yearly_img, summer_img)

#                 # Upload back to Dropbox
#                 self.upload_image(os.path.join(output_path, summer_file), result_img)

#                 # Remove the file from local directory
#                 os.remove(summer_file)
