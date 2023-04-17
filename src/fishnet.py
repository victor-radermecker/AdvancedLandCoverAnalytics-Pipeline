import geopandas as gpd
from shapely.geometry import Polygon
from pyproj import Geod
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from shapely.geometry import MultiPolygon


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

    def create_fishnet(self):
        # Load the shapefile
        self.tx = gpd.read_file(self.shapefile_path)

        # Convert tile size from miles to degrees
        tile_size = self.miles_to_degrees(self.tile_size_miles, self.tx.centroid.y)

        # Calculate the number of rows and columns in the fishnet
        xmin, ymin, xmax, ymax = self.tx.total_bounds
        x_size = xmax - xmin
        y_size = ymax - ymin
        self.num_cols = round(x_size / tile_size)
        self.num_rows = round(y_size / tile_size)

        # Create the fishnet polygons
        fishnet_polys = []
        for i in tqdm(range(self.num_rows)):
            for j in range(self.num_cols):
                # Calculate the coordinates of the fishnet cell corners
                x_left = xmin + j * tile_size
                x_right = xmin + (j + 1) * tile_size
                y_bottom = ymin + i * tile_size
                y_top = ymin + (i + 1) * tile_size

                # Create a polygon for the fishnet cell
                poly = Polygon(
                    [
                        (x_left, y_bottom),
                        (x_right, y_bottom),
                        (x_right, y_top),
                        (x_left, y_top),
                    ]
                )
                fishnet_polys.append(poly)

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

    def plot_fishnet(self):
        # plot the fishnet
        fig, ax = plt.subplots(figsize=(10, 10))
        self.tx.plot(ax=ax, color="white", edgecolor="black")
        self.fishnet.plot(ax=ax, color="none", edgecolor="red")
        plt.show()

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
        filtered_fishnet = self.fishnet[self.fishnet.intersects(bounding_box)]

        return filtered_fishnet

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

    def batch(self, ncols, nrows):
        """
        Group the fishnet tiles into rectangular batches of the same size.

        Parameters:
        ncols (int): The number of columns in each batch.
        nrows (int): The number of rows in each batch.

        Returns:
        GeoDataFrame: A new GeoDataFrame with a less granular fishnet containing all the previous smaller tiles.
        """
        # Calculate the number of original fishnet rows and columns
        xmin, ymin, xmax, ymax = self.fishnet.total_bounds
        x_size = xmax - xmin
        y_size = ymax - ymin

        # Check if ncols and nrows are valid factors of the original fishnet's dimensions
        if self.num_cols % ncols != 0 or self.num_rows % nrows != 0:
            raise ValueError(
                "The provided ncols and nrows are not valid factors for the original fishnet's dimensions."
            )

        # Calculate the number of batch rows and columns
        num_batch_cols = self.num_cols // ncols
        num_batch_rows = self.num_rows // nrows

        # Initialize a list of empty MultiPolygons for each batch cell
        batch_polys = [MultiPolygon() for _ in range(num_batch_rows * num_batch_cols)]

        # Iterate through the smaller tiles in the fishnet
        for tile in tqdm(self.fishnet.itertuples()):
            # Calculate the batch cell indices corresponding to the current tile
            tile_xmin, tile_ymin, tile_xmax, tile_ymax = tile.geometry.bounds
            col_idx = int((tile_xmin - xmin) // (self.tile_size_miles * ncols))
            row_idx = int((tile_ymin - ymin) // (self.tile_size_miles * nrows))

            # Calculate the batch cell index in the batch_polys list
            batch_idx = row_idx * num_batch_cols + col_idx

            # Update the MultiPolygon for the corresponding batch cell with the current tile
            batch_polys[batch_idx] = batch_polys[batch_idx].union(tile.geometry)

        # Create a GeoDataFrame from the batch polygons
        batch_gdf = gpd.GeoDataFrame(
            {"id": range(len(batch_polys)), "geometry": batch_polys},
            crs=self.fishnet.crs,
        )

        return batch_gdf
