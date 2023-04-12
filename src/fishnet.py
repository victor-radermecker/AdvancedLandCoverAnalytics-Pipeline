import geopandas as gpd
from shapely.geometry import Polygon
from pyproj import Geod
from tqdm import tqdm
import matplotlib.pyplot as plt


class Fishnet:
    def __init__(self, shapefile_path, tile_size_miles, overlay_method):
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

    def create_fishnet(self):
        # Load the shapefile
        self.tx = gpd.read_file(self.shapefile_path)

        # Convert tile size from miles to degrees
        tile_size = self.miles_to_degrees(self.tile_size_miles, self.tx.centroid.y)

        # Calculate the number of rows and columns in the fishnet
        xmin, ymin, xmax, ymax = self.tx.total_bounds
        x_size = xmax - xmin
        y_size = ymax - ymin
        num_cols = round(x_size / tile_size)
        num_rows = round(y_size / tile_size)

        # Create the fishnet polygons
        fishnet_polys = []
        for i in tqdm(range(num_rows)):
            for j in range(num_cols):
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
