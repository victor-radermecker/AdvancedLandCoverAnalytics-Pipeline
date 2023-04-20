import geopandas as gpd
from shapely.geometry import Polygon
from pyproj import Geod
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import math
from shapely.geometry import box


class Fishnet:
    def __init__(self, shapefile_path, tile_size_miles, overlay_method, clip=True):
        """
        Initializes a Fishnet object with the given shapefile path and tile size.

        Parameters:
        shapefile_path (str): The file path to the shapefile to use as the base geometry for the fishnet.
        tile_size_miles (float): The size of each tile in miles.
        overlay_method (str): The overlay method to use when clipping the fishnet to the shapefile. (Intersection or Union)

        Returns:
        Fishnet: A Fishnet object with the given shapefile path and tile size.
        """
        self.shapefile_path = shapefile_path
        self.tile_size_miles = tile_size_miles
        self.overlay_method = overlay_method
        self.clip = clip

    # -------------------------------------------------------------------------- #
    #                         Generate fishnet                                   #
    # -------------------------------------------------------------------------- #

    def create_fishnet(self):
        # Load the shapefile
        self.tx = gpd.read_file(self.shapefile_path)

        # Convert tile size from miles to degrees
        self.tile_size_degrees = self.miles_to_degrees(
            self.tile_size_miles, self.tx.centroid.y
        )

        # Calculate the number of rows and columns in the fishnet
        self.xmin, self.ymin, self.xmax, self.ymax = self.tx.total_bounds
        self.x_size = self.xmax - self.xmin
        self.y_size = self.ymax - self.ymin
        self.num_cols = math.ceil(self.x_size / self.tile_size_degrees)
        self.num_rows = math.ceil(self.y_size / self.tile_size_degrees)

        # Create the fishnet polygons
        fishnet_polys = []
        for i in tqdm(range(self.num_rows)):
            for j in range(self.num_cols):
                # Calculate the coordinates of the fishnet cell corners
                x_min = self.xmin + j * self.tile_size_degrees
                x_max = x_min + self.tile_size_degrees
                y_max = self.ymax - i * self.tile_size_degrees
                y_min = y_max - self.tile_size_degrees
                tile_geom = box(x_min, y_min, x_max, y_max)
                fishnet_polys.append(tile_geom)

        # Create a GeoDataFrame from the fishnet polygons
        print("Generating polygons...")
        self.fishnet = gpd.GeoDataFrame(
            {"id": range(len(fishnet_polys)), "geometry": fishnet_polys},
            crs=self.tx.crs,
        )

        if self.clip:
            # Clip the fishnet to the Texas boundary
            print("Cliping fishinet to boundaries...")
            self.fishnet = gpd.overlay(self.fishnet, self.tx, how=self.overlay_method)

        print("Success. Fishnet created.")

        return self.fishnet

    def filter_fishnet_by_bbox(self, bbox):
        """
        Filter the fishnet to keep only the bounding boxes present within the larger bounding box.

        Parameters:
        bbox (tuple): A tuple of (xmin, ymin, xmax, ymax) representing the larger bounding box.

        Returns:
        GeoDataFrame: A filtered GeoDataFrame containing only the bounding boxes within the larger bounding box.
        """
        xmin, ymin, xmax, ymax = bbox
        bounding_box = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
        self.filtered_fishnet = self.fishnet[self.fishnet.intersects(bounding_box)]
        self.filtered_batches = self.batches[self.batches.intersects(bounding_box)]

    # -------------------------------------------------------------------------- #
    #                              Batches                                       #
    # -------------------------------------------------------------------------- #

    def batch(self, batch_tile_size):
        """
        Divide the fishnet into batch tiles of the specified size in miles, assign each fishnet tile to its corresponding
        batch tile, and return the batched fishnet as a GeoDataFrame.

        Parameters:
        batch_tile_size (float): The size of each batch tile in miles.

        Returns:
        GeoDataFrame: A GeoDataFrame with the same geometry as the fishnet, but with an additional column indicating the
        batch tile ID of each fishnet tile.
        """
        # Check that batch tile size is a multiple of tile size
        if batch_tile_size % self.tile_size_miles != 0:
            raise ValueError("Batch tile size must be a multiple of tile size.")
        else:
            self.batch_tile_size = batch_tile_size

        # Convert batch tile size from miles to degrees
        self.batch_tile_size_degrees = self.miles_to_degrees(
            self.batch_tile_size, self.tx.centroid.y
        )

        # Calculate the number of rows and columns in the batched fishnet
        self.batch_cols = math.ceil(self.x_size / self.batch_tile_size_degrees)
        self.batch_rows = math.ceil(self.y_size / self.batch_tile_size_degrees)

        # Create a dictionary to store the batch tile ID of each fishnet tile
        batch_dict = []

        # Iterate over the fishnet tiles and assign each one to its corresponding batch tile
        for i, row in tqdm(self.fishnet.iterrows(), total=self.fishnet.shape[0]):
            col_idx = i % self.num_cols
            row_idx = i // self.num_cols
            batch_col_idx = col_idx // (
                self.batch_tile_size_degrees / self.tile_size_degrees
            )
            batch_row_idx = row_idx // (
                self.batch_tile_size_degrees / self.tile_size_degrees
            )
            batch_id = int(batch_row_idx * self.batch_cols + batch_col_idx)
            batch_dict.append(batch_id)

        # Create a new GeoDataFrame with the batch IDs
        self.fishnet["batch_id"] = pd.Series(batch_dict)

        # Create a new GeoDataFrame with the batch geometries
        batch_geoms = []
        for i in tqdm(range(self.batch_rows)):
            for j in range(self.batch_cols):
                x_min = self.xmin + j * self.batch_tile_size_degrees
                x_max = x_min + self.batch_tile_size_degrees
                y_max = self.ymax - i * self.batch_tile_size_degrees
                y_min = y_max - self.batch_tile_size_degrees
                batch_geom = box(x_min, y_min, x_max, y_max)
                batch_geoms.append(batch_geom)
        batches = gpd.GeoDataFrame(
            {
                "batch_id": range(self.batch_rows * self.batch_cols),
                "geometry": batch_geoms,
            },
            crs=self.fishnet.crs,
        )
        self.batches = batches

    # -------------------------------------------------------------------------- #
    #                               Utils                                       #
    # -------------------------------------------------------------------------- #

    def save(self, file_path):
        """
        Save the Fishnet object to a file using pickle.

        Parameters:
        file_path (str): The file path to save the Fishnet object.
        """
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, file_path):
        """
        Load a Fishnet object from a file using pickle.

        Parameters:
        file_path (str): The file path of the saved Fishnet object.

        Returns:
        Fishnet: The loaded Fishnet object.
        """
        with open(file_path, "rb") as file:
            return pickle.load(file)

    def plot_fishnet(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        self.tx.plot(ax=ax, color="white", edgecolor="black")
        # Plot the fishnet tiles
        self.fishnet.plot(ax=ax, color="none", edgecolor="red")
        # Plot the batch tiles
        self.batches.plot(ax=ax, color="none", edgecolor="darkgreen", linewidth=3)
        plt.show()

    def plot_filtered_fishnet(self, zoom=False):
        # check if self.filtered_fishnet exists
        if not hasattr(self, "filtered_fishnet"):
            print(
                "No filtered fishnet found. Please run filter_fishnet_by_bbox() first."
            )
        fig, ax = plt.subplots(figsize=(10, 10))
        self.tx.plot(ax=ax, color="white", edgecolor="black")
        # Plot the fishnet tiles
        self.filtered_fishnet.plot(ax=ax, color="none", edgecolor="red")
        # Plot the batch tiles
        self.filtered_batches.plot(
            ax=ax, color="none", edgecolor="darkgreen", linewidth=3
        )
        if zoom:
            ax.set_xlim(
                self.filtered_batches.total_bounds[0],
                self.filtered_batches.total_bounds[2],
            )
            ax.set_ylim(
                self.filtered_batches.total_bounds[1],
                self.filtered_batches.total_bounds[3],
            )
        plt.show()

    def miles_to_degrees(self, miles, latitude):
        """
        Convert a distance in miles to decimal degrees at a specific latitude.

        This function uses the pyproj.Geod class to perform more accurate distance
        calculations using a geodetic coordinate system (WGS84).

        Parameters:
        miles (float): The distance in miles to be converted.
        latitude (float): The latitude at which the conversion is to be performed.

        Returns:
        float: The distance in decimal degrees corresponding to the input distance in miles.
        """
        geod = Geod(ellps="WGS84")
        meters = miles * 1609.34
        new_longitude, _, _ = geod.fwd(0, latitude, 90, meters)
        return abs(new_longitude)
