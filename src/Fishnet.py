import geopandas as gpd
from shapely.geometry import Polygon

# from pyproj import Geod
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import math
from prettytable import PrettyTable

# import matplotlib as mpl
# import numpy as np
# import seaborn as sns
import folium
from tqdm.contrib import tzip

# import branca.colormap as cm
from shapely.geometry import box
from tqdm import tqdm


class Fishnet:
    def __init__(
        self,
        tile_size_miles,
        coordinates=None,
        shapefile_path=None,
        clip=False,
        overlay_method=None,
    ):
        """
        Initializes a Fishnet object with the given shapefile path and tile size.

        Parameters:
        shapefile_path (str): The file path to the shapefile to use as the base geometry for the fishnet.
        tile_size_miles (float): The size of each tile in miles.
        overlay_method (str): The overlay method to use when clipping the fishnet to the shapefile. (Intersection or Union)

        Returns:
        Fishnet: A Fishnet object with the given shapefile path and tile size.
        """
        self.tile_width_miles = tile_size_miles
        self.tile_height_miles = tile_size_miles

        # The user may provid only a Start point or only a Shapefile
        if coordinates is not None and shapefile_path is None:
            self.init_with_coordinates(coordinates)
        elif shapefile_path is not None and coordinates is None:
            self.init_with_shapefile(shapefile_path, overlay_method, clip)
        else:
            raise Exception(
                "You must provide either a start point or a shapefile path to initialize the fishnet."
            )

    # -------------------------------------------------------------------------- #
    #                         Generate fishnet                                   #
    # -------------------------------------------------------------------------- #

    def init_with_coordinates(self, coordinates):
        self.shapefile = False
        self.xmin, self.ymin, self.xmax, self.ymax = coordinates
        self.crs = "EPSG:4326"
        print("Using Coordinates to initialize fishnet.")

    def init_with_shapefile(self, shapefile_path, overlay_method, clip):
        self.shapefile = True
        self.shapefile_path = shapefile_path
        self.overlay_method = overlay_method
        self.clip = clip
        self.tx = gpd.read_file(self.shapefile_path)
        self.xmin, self.ymin, self.xmax, self.ymax = self.tx.total_bounds
        self.crs = self.tx.crs
        print("Using Shapefile to initialize fishnet.")

    def create_fishnet(self):
        # Initialize the fishnet parameters
        self.fishnet_width_degrees = self.xmax - self.xmin
        self.fishnet_height_degrees = self.ymax - self.ymin
        _, self.fishnet_width_miles = self.lat_lon_change_to_miles(
            0, self.xmax - self.xmin, self.ymin
        )
        self.fishnet_height_miles, _ = self.lat_lon_change_to_miles(
            self.ymax - self.ymin, 0, self.ymin
        )

        # Convert tile size from miles to degrees
        _, self.tile_width_degrees = self.miles_to_lat_lon_change(
            self.ymin, self.xmin, self.tile_width_miles, 90
        )
        self.tile_height_degrees, _ = self.miles_to_lat_lon_change(
            self.ymin, self.xmin, self.tile_height_miles, 0
        )

        # Calculate the number of rows and columns in the fishnet
        self.fishnet_cols = math.ceil(
            self.fishnet_width_degrees / self.tile_width_degrees
        )
        self.fishnet_rows = math.ceil(
            self.fishnet_height_degrees / self.tile_height_degrees
        )

        # Create the fishnet polygons
        fishnet_polys = []
        for i in tqdm(range(self.fishnet_rows)):
            for j in range(self.fishnet_cols):
                # Calculate the coordinates of the fishnet cell corners
                x_min = self.xmin + j * self.tile_width_degrees
                x_max = x_min + self.tile_width_degrees
                y_max = self.ymax - i * self.tile_height_degrees
                y_min = y_max - self.tile_height_degrees
                tile_geom = box(x_min, y_min, x_max, y_max)
                fishnet_polys.append(tile_geom)

        # Create a GeoDataFrame from the fishnet polygons
        print("Generating polygons...")
        self.fishnet = gpd.GeoDataFrame(
            {"id": range(len(fishnet_polys)), "geometry": fishnet_polys},
            crs=self.crs,
        )

        if self.clip:
            # Clip the fishnet to the Shapefile boundary
            print("Cliping fishinet to boundaries...")
            self.fishnet = gpd.overlay(self.fishnet, self.tx, how=self.overlay_method)

        print("Success. Fishnet created.")
        self.fishnet_info()

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
        if batch_tile_size % self.tile_width_miles != 0:
            raise ValueError("Batch tile size must be a multiple of tile size.")
        else:
            self.batch_tile_size = batch_tile_size
            print("Batching fishnet... \n")

        # Convert batch tile size from miles to degrees
        self.batch_width_miles = self.batch_tile_size
        self.batch_height_miles = self.batch_tile_size
        _, self.batch_width_degrees = self.miles_to_lat_lon_change(
            self.ymin, self.xmin, self.batch_tile_size, 90
        )
        self.batch_height_degrees, _ = self.miles_to_lat_lon_change(
            self.ymin, self.xmin, self.batch_tile_size, 0
        )

        # Calculate the number of rows and columns in the batched fishnet
        self.batch_cols = math.ceil(
            self.fishnet_width_degrees / self.batch_width_degrees
        )
        self.batch_rows = math.ceil(
            self.fishnet_height_degrees / self.batch_height_degrees
        )

        # Calculate batch_id for each tile in fishnet
        self.fishnet["batch_id"] = self._create_batches()
        self.fishnet["batch_id"] = self.fishnet["batch_id"].astype(int)

        # Create batch geometry GeoDataFrame
        self.batches = self._create_batch_geometries()
        print("Success. Fishnet batched.")
        self.batch_info()

    def _create_batches(self):
        self.nbr_tiles_per_batch = self.batch_width_miles / self.tile_width_miles
        tile_row, tile_col = np.divmod(
            np.arange(self.fishnet_cols * self.fishnet_rows), self.fishnet_cols
        )
        batch_id = (tile_row // self.nbr_tiles_per_batch) * self.batch_cols + (
            tile_col // self.nbr_tiles_per_batch
        )
        return batch_id.tolist()

    def _create_batch_geometries(self):
        min_xs = (
            self.xmin + np.arange(self.batch_cols) * self.batch_width_degrees
        )  # minimum x values (longitude) for each column in the batch grid
        max_ys = (
            self.ymax - np.arange(self.batch_rows) * self.batch_height_degrees
        )  # maximum y values (latitude) for each row in the batch grid
        geometries = [
            box(
                x_min,
                max_ys[i],
                x_min + self.batch_width_degrees,
                max_ys[i] - self.batch_height_degrees,
            )
            for i in range(self.batch_rows)
            for x_min in min_xs
        ]
        return gpd.GeoDataFrame(
            {
                "batch_id": range(self.batch_rows * self.batch_cols),
                "geometry": geometries,
            },
            crs=self.crs,
        )

        # # Create a dictionary to store the batch tile ID of each fishnet tile
        # batch_dict = []

        # # Iterate over the fishnet tiles and assign each one to its corresponding batch tile
        # for i, row in tqdm(self.fishnet.iterrows(), total=self.fishnet.shape[0]):
        #     col_idx = i % self.fishnet_cols  # col index within the fishnet
        #     row_idx = i // self.fishnet_cols  # row index within the fishnet
        #     batch_col_idx = col_idx // (
        #         self.batch_width_degrees / self.tile_width_degrees
        #     )
        #     batch_row_idx = row_idx // (
        #         self.batch_height_degrees / self.tile_height_degrees
        #     )
        #     batch_id = int(batch_row_idx * self.batch_cols + batch_col_idx)
        #     batch_dict.append(batch_id)

        # # Create a new GeoDataFrame with the batch geometries
        # batch_geoms = []
        # for i in tqdm(range(self.batch_rows)):
        #     for j in range(self.batch_cols):
        #         x_min = self.xmin + j * self.batch_delta_lon
        #         x_max = x_min + self.batch_delta_lon
        #         y_max = self.ymax - i * self.batch_delta_lat
        #         y_min = y_max - self.batch_delta_lat
        #         batch_geom = box(x_min, y_min, x_max, y_max)
        #         batch_geoms.append(batch_geom)

        # self.batches = gpd.GeoDataFrame(
        #     {
        #         "batch_id": range(self.batch_rows * self.batch_cols),
        #         "geometry": batch_geoms,
        #     },
        #     crs=self.fishnet.crs,
        # )

        # # Create a new GeoDataFrame with the batch IDs
        # self.fishnet["batch_id"] = pd.Series(batch_dict)

    # -------------------------------------------------------------------------- #
    #                          Tile Neighbors                                    #
    # -------------------------------------------------------------------------- #

    def compute_neighbors(self):
        self.neighbors = {}

        for i in tqdm(
            range(self.num_rows), total=self.num_rows, desc="Computing neighbors..."
        ):
            for j in range(self.num_cols):
                neighbor_indices = [
                    (i + ii, j + jj)
                    for ii in range(-1, 2)
                    for jj in range(-1, 2)
                    if (ii != 0 or jj != 0)
                ]
                neighbor_indices = [
                    (x, y)
                    for x, y in neighbor_indices
                    if x >= 0 and x < self.num_rows and y >= 0 and y < self.num_cols
                ]
                neighbor_ids = [self.row_col_to_id(x, y) for x, y in neighbor_indices]
                self.neighbors[self.row_col_to_id(i, j)] = neighbor_ids

        # add neighbors to fishnet
        self.fishnet["neighbors"] = self.fishnet.apply(
            lambda row: self.neighbors[row["id"]], axis=1
        )

        # add neighbors to filtered fishnet
        self.filtered_fishnet["neighbors"] = self.filtered_fishnet.apply(
            lambda row: self.neighbors[row["id"]], axis=1
        )

        print("All neighbors computed successfully.")

    # -------------------------------------------------------------------------- #
    #                               Plots                                        #
    # -------------------------------------------------------------------------- #

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

    def plot_heatmap(self, feature, filtered, zoom=True):
        if filtered:
            df = self.filtered_fishnet
        else:
            df = self.fishnet

        if feature not in df.columns:
            raise ValueError("The feature is not available in the DataFrame.")
        else:
            df = gpd.GeoDataFrame(df, geometry="geometry", crs={"init": "epsg:4326"})
            geo = {
                "type": "FeatureCollection",
                "features": df.apply(
                    lambda row: {
                        "type": "Feature",
                        "geometry": row["geometry"].__geo_interface__,
                        "properties": {"id": row["id"]},
                    },
                    axis=1,
                ).tolist(),
            }
            avg_lat = df["geometry"].apply(lambda x: x.centroid.y).mean()
            avg_lon = df["geometry"].apply(lambda x: x.centroid.x).mean()
            map = folium.Map(location=[avg_lat, avg_lon], zoom_start=11)

            choropleth = folium.Choropleth(
                geo_data=geo,
                name="choropleth",
                data=df,
                columns=["id", feature],
                key_on="feature.properties.id",
                fill_opacity=0.7,
                fill_color="YlOrRd",
                line_opacity=0.8,
                legend_name=feature,
            )
            choropleth.add_to(map)

            return map

    def plot_neighbor(self):
        # sample one random row from geodataframe
        row = self.fishnet.sample()

        # find coordinates of centroid of geometry
        # find map center
        mean_x = (row["geometry"].bounds["minx"] + row["geometry"].bounds["maxx"]) / 2
        mean_y = (row["geometry"].bounds["miny"] + row["geometry"].bounds["maxy"]) / 2

        # find all neighbors
        neighbors = self.fishnet[self.fishnet["id"].isin(list(row["neighbors"])[0])]

        # create empty map
        m = folium.Map(
            location=[mean_y, mean_x], zoom_start=15, tiles="CartoDB positron"
        )

        # add polygon corresponding to id considered
        m = self.add_polygon_to_map(m, row)

        # add polygons corresponding to neighbors
        for _, r in neighbors.iterrows():
            m = self.add_polygon_to_map(m, r)

        # show map
        display(m)

    def add_polygon_to_map(self, m, r):
        # given map object and row from geodataframe, add to map the corresponding geometry
        sim_geo = gpd.GeoSeries(r["geometry"]).simplify(tolerance=0.001)
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(
            data=geo_j, style_function=lambda x: {"fillColor": "orange"}
        )
        geo_j.add_to(m)

        return m

    # -------------------------------------------------------------------------- #
    #                               Utils                                        #
    # -------------------------------------------------------------------------- #

    def fishnet_info(self):
        print("\n Fishnet Object has the following attributes: \n")
        table = PrettyTable()
        table.field_names = ["Metric", "Degrees", "Miles"]
        table.add_row(
            ["Fishnet Width", self.fishnet_width_degrees, self.fishnet_width_miles]
        )
        table.add_row(
            ["Fishnet Height", self.fishnet_height_degrees, self.fishnet_height_miles]
        )
        table.add_row(["Tiles Width", self.tile_width_degrees, self.tile_width_miles])
        table.add_row(
            ["Tiles Height", self.tile_height_degrees, self.tile_height_miles]
        )
        print(table)

    def batch_info(self):
        print("\nFishnet Batch has the following attributes: \n")
        print(
            "Number of tiles per batch: ",
            int(self.batch_width_miles / self.tile_width_miles),
        )
        table = PrettyTable()
        table.field_names = ["Metric", "Tiles", "Batches"]
        table.add_row(["Rows", self.fishnet_rows, self.batch_rows])
        table.add_row(["Columns", self.fishnet_cols, self.batch_cols])
        table.add_row(
            [
                "Cells",
                self.fishnet_rows * self.fishnet_cols,
                self.batch_rows * self.batch_cols,
            ]
        )
        print(table)

        table = PrettyTable()
        table.field_names = ["Metric", "Degrees", "Miles"]
        table.add_row(["Batch Width", self.batch_width_degrees, self.batch_width_miles])
        table.add_row(
            ["Batch Height", self.batch_height_degrees, self.batch_height_miles]
        )
        print(table)

    def info(self):
        self.fishnet_info()
        self.batch_info()

    def save(self, file_path):
        """
        Save the Fishnet object to a file using pickle.

        Parameters:
        file_path (str): The file path to save the Fishnet object.
        """
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    def load(file_path):
        """
        Load a Fishnet object from a file using pickle.

        Parameters:
        file_path (str): The file path of the saved Fishnet object.

        Returns:
        Fishnet: The loaded Fishnet object.
        """
        with open(file_path, "rb") as file:
            return pickle.load(file)

    def compute_difference(self, feature1, feature2, filtered=False, normalize=False):
        if filtered:
            df = self.filtered_fishnet
        else:
            df = self.fishnet

        if feature1 not in df.columns or feature2 not in df.columns:
            raise ValueError(
                "One of the features is not available in the Fishnet object."
            )
        else:
            df[feature1 + "-" + feature2] = df[feature1] - df[feature2]
            if normalize:
                df[feature1 + "-" + feature2] = df[feature1 + "-" + feature2] / 255

    def row_col_to_id(self, i, j):
        return i * self.num_cols + j

    # -------------------------------------------------------------------------- #
    #                          Harvesine Formula                                 #
    # -------------------------------------------------------------------------- #

    def miles_to_lat_lon_change(self, lat, lon, distance_miles, bearing_degrees):
        """
        Calculate the change in latitude and longitude for a given distance and bearing from a starting point.

        The function uses the 'haversine' formula to calculate the change in latitude and longitude
        from the starting point. This method is effective for relatively short distances on the Earth's surface.

        Args:
            lat (float): The latitude of the starting point in degrees.
            lon (float): The longitude of the starting point in degrees.
            distance_miles (float): The distance to travel from the starting point in miles.
            bearing_degrees (float): The direction to travel from the starting point in degrees, where 0 is North
                                     and increases clockwise.

        Returns:
            tuple: A tuple containing two elements:
                   1. The change in latitude from the starting point (float).
                   2. The change in longitude from the starting point (float).

        Notes:
            The Earth is assumed to be perfectly spherical for this calculation, which introduces some error.
            For more accurate results over larger distances, an ellipsoidal model should be used.
        """

        R = 6371  # Earth's radius in kilometers
        distance_km = distance_miles * 1.60934

        lat1_rad = math.radians(lat)
        lon1_rad = math.radians(lon)
        bearing_rad = math.radians(bearing_degrees)

        lat2_rad = math.asin(
            math.sin(lat1_rad) * math.cos(distance_km / R)
            + math.cos(lat1_rad) * math.sin(distance_km / R) * math.cos(bearing_rad)
        )

        lon2_rad = lon1_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(distance_km / R) * math.cos(lat1_rad),
            math.cos(distance_km / R) - math.sin(lat1_rad) * math.sin(lat2_rad),
        )

        lat2 = math.degrees(lat2_rad)
        lon2 = math.degrees(lon2_rad)

        return lat2 - lat, lon2 - lon

    def lat_lon_change_to_miles(self, delta_lat, delta_lon, lat):
        """
        Calculate the distance in miles for a given change in latitude and longitude from a starting latitude.

        The function uses the 'haversine' formula to calculate the distance in miles
        from the change in latitude and longitude.

        Args:
            delta_lat (float): The change in latitude from the starting point in degrees.
            delta_lon (float): The change in longitude from the starting point in degrees.
            lat (float): The latitude of the starting point in degrees.

        Returns:
            float: The distance in miles corresponding to the change in latitude and longitude.

        Notes:
            The Earth is assumed to be perfectly spherical for this calculation, which introduces some error.
            For more accurate results, an ellipsoidal model should be used.
        """
        R = 6371.0
        delta_lat_rad = math.radians(delta_lat)
        delta_lon_rad = math.radians(delta_lon)
        distance_lat_miles = delta_lat_rad * R * 0.621371
        distance_lon_miles = delta_lon_rad * R * math.cos(math.radians(lat)) * 0.621371
        return distance_lat_miles, distance_lon_miles
