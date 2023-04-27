import ee
import geemap
from tqdm import tqdm

ee.Initialize()


class ImageExporter:
    def __init__(self, fishnet, filtered, scale=10):
        self.fh = fishnet
        self.scale = scale
        self.fileFormat = "GeoTIFF"
        self.maxPixels = 1e10
        self.filtered = filtered

        if self.filtered:
            self.fishnet = self.fh.filtered_fishnet
            self.batches = self.fh.filtered_batches
        else:
            self.fishnet = self.fh.fishnet
            self.batches = self.fh.batches

    def set_date_range(self, year):
        startDate = f"{year}-05-01"
        endDate = f"{year}-09-01"

        self.startDate = startDate
        self.endDate = endDate

    def set_folder(self, folder):
        # Google Drive Folder that the export will reside in
        self.folder = folder

    def export_images(self, region):
        for i in tqdm(range(len(self.batches))):
            batch = self.batches.iloc[i]
            geom_region = ee.Geometry.Rectangle(region)
            batch_region = ee.Geometry.Rectangle(batch["geometry"].bounds)
            intersection = geom_region.intersection(batch_region)
            if intersection.is_empty().getInfo():
                continue

            landcover = geemap.dynamic_world(
                intersection, self.startDate, self.endDate, return_type="visualize"
            )

            # Save the image
            export_params = {
                "image": landcover,
                "description": f'landcover_batchID_{batch["batch_id"]}',
                "folder": self.folder,  # Google Drive folder name
                "scale": self.scale,  # Resolution in meters
                "region": intersection,
                "fileFormat": self.fileFormat,
                "maxPixels": self.maxPixels,  # Increase this value if you encounter an error due to the pixel limit
            }

            export_task = ee.batch.Export.image.toDrive(**export_params)
            export_task.start()
