import ee
from tqdm import tqdm
import GeemapUtils as geemap

ee.Initialize()


class ImageExporter:
    def __init__(self, fishnet, filtered, scale=10, source = "GOOGLE/DYNAMICWORLD/V1"):  # "USGS/NLCD_RELEASES/2019_REL/NLCD" for NLCD
        self.fh = fishnet
        self.scale = scale
        self.fileFormat = "GeoTIFF"
        self.maxPixels = 1e10
        self.filtered = filtered
        self.source = source

        if self.filtered:
            self.fishnet = self.fh.filtered_fishnet
            self.batches = self.fh.filtered_batches
        else:
            self.fishnet = self.fh.fishnet
            self.batches = self.fh.batches

    def set_date_range(self, startYear, endYear, startMonth, endMonth):
        startDate = f"{startYear}-{startMonth}-01"
        endDate = f"{endYear}-{endMonth}-01"

        self.startDate = startDate
        self.endDate = endDate
        self.year = startYear

    def set_folder(self, folder):
        # Google Drive Folder that the export will reside in
        self.folder = folder

    def export_images(self):
        for i in tqdm(range(len(self.batches))):
            batch = self.batches.iloc[i]
            batch_region = ee.Geometry.Rectangle(batch["geometry"].bounds)

            if self.source == "GOOGLE/DYNAMICWORLD/V1":
                landcover = geemap.dynamic_world(
                    batch_region, self.startDate, self.endDate, return_type="visualize", source = self.source
                )
            elif self.source == "USGS/NLCD_RELEASES/2019_REL/NLCD":
                
                nlcd = ee.ImageCollection("USGS/NLCD_RELEASES/2019_REL/NLCD").filter(ee.Filter.eq('system:index', f'{self.year}')).first()
                landcover = nlcd.clip(batch_region).select(['landcover'])

            # Save the image
            export_params = {
                "image": landcover,
                "description": f'landcover_batchID_{batch["batch_id"]}',
                "folder": self.folder,  # Google Drive folder name
                "scale": self.scale,  # Resolution in meters
                "region": batch_region,
                "fileFormat": self.fileFormat,
                "maxPixels": self.maxPixels,  # Increase this value if you encounter an error due to the pixel limit
            }

            export_task = ee.batch.Export.image.toDrive(**export_params)
            export_task.start()
