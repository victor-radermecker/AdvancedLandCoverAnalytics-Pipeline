###################################################################################################
#                                              ATTRIBUTES                                      ####
###################################################################################################

TILE_SIZE_MILES = 0.25
BATCH_SIZE_MILES = 4
FILTER_REGION = [-95.799944, 29.374853, -95.028636, 29.795492]
IMG_DIR = "./Images2/"

# import modules
import sys
import warnings

warnings.filterwarnings("ignore")
import ee
from tqdm import tqdm

# import local modules
sys.path.append("./src")
from Fishnet import Fishnet
from ImageExporter import ImageExporter
from ImageCorrector import ImageCorrector
from ImageProcessor import ImageProcessor

# Generate a fishnet using a shapefile
fc = Fishnet(
    tile_size_miles=TILE_SIZE_MILES,
    coordinates=None,
    shapefile_path="./Gis/Texas_State_Boundary/State.shp",
    clip=False,
    overlay_method=None,
)
fc.create_fishnet()
fc.batch(BATCH_SIZE_MILES)
fc.filter_fishnet_by_bbox(FILTER_REGION)
fc.compute_neighbors()
corrector = ImageCorrector(IMG_DIR)
corrector.correct_images()
img_process = ImageProcessor(
    fc,
    filtered=True,
)
img_process.assign_fishnet_tiles_to_pixels(
    IMG_DIR + f"export_{2016}/Final/", "landcover_batchID"
)
for year in tqdm([2016, 2017, 2018, 2019, 2020, 2021, 2022]):
    # Compute urbanization rate
    img_process.compute_mean_tile_entropy_urbanization(
        IMG_DIR + f"/export_{year}/Final/",
        "landcover_batchID",
        f"MeanPixel_{year}",
        f"Entropy_{year}",
    )
for year in [2017, 2018, 2019, 2020, 2021, 2022]:
    fc.compute_difference(
        f"MeanPixel_{year}", f"MeanPixel_{year-1}", filtered=True, normalize=True
    )

# Save result
fc.save("./Gis/Fishnet/fishnet_quarter_mile.pkl")
