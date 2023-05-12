import geopandas as gpd
from shapely.geometry import Polygon
from pyproj import Geod
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import math
import matplotlib as mpl
import numpy as np
import seaborn as sns
import folium
import branca.colormap as cm
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
        centroid = self.tx.unary_union.centroid
        _, self.delta_lon = self.miles_to_lat_lon_change(
            centroid.y, centroid.x, self.tile_size_miles, 90
        )
        self.delta_lat, _ = self.miles_to_lat_lon_change(
            centroid.y, centroid.x, self.tile_size_miles, 0
        )

        # Calculate the number of rows and columns in the fishnet
        self.xmin, self.ymin, self.xmax, self.ymax = self.tx.total_bounds
        self.x_size = self.xmax - self.xmin
        self.y_size = self.ymax - self.ymin
        self.num_cols = math.ceil(self.x_size / self.delta_lon)
        self.num_rows = math.ceil(self.y_size / self.delta_lat)

        # Create the fishnet polygons
        fishnet_polys = []
        for i in tqdm(range(self.num_rows)):
            for j in range(self.num_cols):
                # Calculate the coordinates of the fishnet cell corners
                x_min = self.xmin + j * self.delta_lon
                x_max = x_min + self.delta_lon
                y_max = self.ymax - i * self.delta_lat
                y_min = y_max - self.delta_lat
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
        centroid = self.tx.unary_union.centroid
        _, self.batch_delta_lon = self.miles_to_lat_lon_change(
            centroid.y, centroid.x, self.batch_tile_size, 90
        )
        self.batch_delta_lat, _ = self.miles_to_lat_lon_change(
            centroid.y, centroid.x, self.batch_tile_size, 0
        )

        # Calculate the number of rows and columns in the batched fishnet
        self.batch_cols = math.ceil(self.x_size / self.batch_delta_lon)
        self.batch_rows = math.ceil(self.y_size / self.batch_delta_lat)

        # Create a dictionary to store the batch tile ID of each fishnet tile
        batch_dict = []

        # Iterate over the fishnet tiles and assign each one to its corresponding batch tile
        for i, row in tqdm(self.fishnet.iterrows(), total=self.fishnet.shape[0]):
            col_idx = i % self.num_cols
            row_idx = i // self.num_cols
            batch_col_idx = col_idx // (self.batch_delta_lon / self.delta_lon)
            batch_row_idx = row_idx // (self.batch_delta_lat / self.delta_lat)
            batch_id = int(batch_row_idx * self.batch_cols + batch_col_idx)
            batch_dict.append(batch_id)

        # Create a new GeoDataFrame with the batch geometries
        batch_geoms = []
        for i in tqdm(range(self.batch_rows)):
            for j in range(self.batch_cols):
                x_min = self.xmin + j * self.batch_delta_lon
                x_max = x_min + self.batch_delta_lon
                y_max = self.ymax - i * self.batch_delta_lat
                y_min = y_max - self.batch_delta_lat
                batch_geom = box(x_min, y_min, x_max, y_max)
                batch_geoms.append(batch_geom)

        self.batches = gpd.GeoDataFrame(
            {
                "batch_id": range(self.batch_rows * self.batch_cols),
                "geometry": batch_geoms,
            },
            crs=self.fishnet.crs,
        )

        # Create a new GeoDataFrame with the batch IDs
        self.fishnet["batch_id"] = pd.Series(batch_dict)

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

    def load(self, file_path):
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

    # def plot_heatmap(self, feature, filtered, zoom=True):
    #     if filtered:
    #         df = self.filtered_fishnet
    #     else:
    #         df = self.fishnet

    #     if feature not in df.columns:
    #         raise ValueError("The feature is not available in the Fishnet object.")
    #     else:
    #         fig, ax = plt.subplots(figsize=(10, 18))
    #         self.tx.plot(ax=ax, color="white", edgecolor="black")

    #         # Plot the heatmap using imshow
    #         sc = ax.imshow(np.array(df[feature]), cmap="inferno", origin="lower")

    #         if zoom:
    #             ax.set_xlim(df.total_bounds[0], df.total_bounds[2])
    #             ax.set_ylim(df.total_bounds[1], df.total_bounds[3])

    #     # Add a horizontal colorbar below the plot
    #     cax = fig.add_axes([0.1, 0.15, 0.8, 0.03])
    #     cbar = plt.colorbar(sc, cax=cax, orientation="horizontal")

    #     plt.show()

    # def plot_heatmap(self, feature, filtered, zoom=True):
    #     if filtered:
    #         df = self.filtered_fishnet
    #     else:
    #         df = self.fishnet

    #     if feature not in df.columns:
    #         raise ValueError("The feature is not available in the Fishnet object.")
    #     else:
    #         fig, ax = plt.subplots(figsize=(10, 18))
    #         self.tx.plot(ax=ax, color="white", edgecolor="black")
    #         # sc = df.plot(
    #         #     ax=ax,
    #         #     column=feature,
    #         #     cmap="inferno",
    #         #     edgecolor="gray",
    #         #     linewidth=0.5,
    #         # )
    #         sc = ax.imshow(np.array(df[feature]), cmap="inferno", origin="lower")

    #         if zoom:
    #             ax.set_xlim(df.total_bounds[0], df.total_bounds[2])
    #             ax.set_ylim(df.total_bounds[1], df.total_bounds[3])

    #     # Add a horizontal colorbar below the plot
    #     cax = fig.add_axes([0.1, 0.15, 0.8, 0.03])
    #     cbar = plt.colorbar(sc, cax=cax, orientation="horizontal")

    #     plt.show()

    # def plot_heatmap(self, feature, filtered, zoom=True):
    #     if filtered:
    #         df = self.filtered_fishnet
    #     else:
    #         df = self.fishnet

    #     if feature not in df.columns:
    #         raise ValueError("The feature is not available in the Fishnet object.")
    #     else:
    #         fig, ax = plt.subplots(figsize=(10, 18))
    #         self.tx.plot(ax=ax, color="white", edgecolor="black")
    #         df.plot(
    #             ax=ax,
    #             column=feature,
    #             cmap="YlGn",
    #             edgecolor="gray",
    #             linewidth=0.5,
    #             legend=True,
    #         )

    #         # Add title and axis labels
    #         plt.title("Choropleth Map")
    #         plt.xlabel("Longitude")
    #         plt.ylabel("Latitude")

    #         # Place the legend under the plot and rotate it by 90 degrees
    #         leg = ax.get_legend()
    #         leg.set_bbox_to_anchor((0, 0, 1, 0.15))
    #         leg.set_title("Feature")
    #         plt.setp(leg.get_title(), fontsize=12)
    #         plt.setp(leg.get_texts(), fontsize=10)
    #         leg._loc = 2
    #         leg.set_frame_on(True)
    #         leg.set_edgecolor("black")
    #         leg.set_linewidth(0.5)
    #         leg.set_alpha(0.7)

    #         if zoom:
    #             ax.set_xlim(df.total_bounds[0], df.total_bounds[2])
    #             ax.set_ylim(df.total_bounds[1], df.total_bounds[3])

    #     plt.show()

    # def plot_heatmap(self, feature, filtered, zoom=True):
    #     if filtered:
    #         df = self.filtered_fishnet
    #     else:
    #         df = self.fishnet

    #     if feature not in df.columns:
    #         raise ValueError("The feature is not available in the DataFrame.")
    #     else:
    #         df = gpd.GeoDataFrame(df, geometry="geometry", crs={"init": "epsg:4326"})
    #         geo = gpd.GeoSeries(df.set_index("batch_id")["geometry"]).to_json()
    #         avg_lat = df["geometry"].apply(lambda x: x.centroid.y).mean()
    #         avg_lon = df["geometry"].apply(lambda x: x.centroid.x).mean()
    #         map = folium.Map(location=[avg_lat, avg_lon], zoom_start=11)

    #         choropleth = folium.Choropleth(
    #             geo_data=geo,
    #             name="choropleth",
    #             data=df,
    #             columns=["batch_id", feature],
    #             key_on="feature.id",
    #             fill_opacity=0.7,
    #             fill_color="YlOrRd",
    #             line_opacity=0.8,
    #             legend_name=feature,
    #         )
    #         choropleth.add_to(map)

    #         return map

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

    def miles_to_lat_lon_change(self, lat, lon, distance_miles, bearing_degrees):
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
