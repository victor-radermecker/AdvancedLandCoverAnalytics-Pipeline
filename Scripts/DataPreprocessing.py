###################################################################################################
#                                              ATTRIBUTES                                      ####
###################################################################################################

TILE_SIZE_MILES = 0.25
BATCH_SIZE_MILES = 16
FILTER_REGION = [-99.13, 28.91, -94.29, 31.1]
IMG_DIR = "./Images3/"
SAVE_DIR = "./Outputs"
SHAPEFILE_PATH = "./Gis/Texas_State_Boundary/State.shp"

# import modules
import sys
import warnings
import os

warnings.filterwarnings("ignore")
import ee
from tqdm import tqdm
import matplotlib.pyplot as plt

# import local modules
from Fishnet import Fishnet
from ImageExporter import ImageExporter
from ImageCorrector import ImageCorrector
from ImageProcessor import ImageProcessor

# check the directories exist
if not os.path.exists(IMG_DIR):
    print(f"Creating directory {IMG_DIR}.")
    os.makedirs(IMG_DIR)
if not os.path.exists(SAVE_DIR):
    print(f"Creating directory {SAVE_DIR}.")
    os.makedirs(SAVE_DIR)

###################################################################################################
#                                         FISHNET GENERATION                                   ####
###################################################################################################

# Generate a fishnet using a shapefile
fc = Fishnet(
    tile_size_miles=TILE_SIZE_MILES,
    coordinates=None,
    shapefile_path=SHAPEFILE_PATH,
    clip=False,
    overlay_method=None,
)
fc.create_fishnet()
fc.batch(BATCH_SIZE_MILES)
fc.filter_fishnet_by_bbox(FILTER_REGION)
fc.compute_neighbors()

###################################################################################################
#                                  IMAGE CORRECTOR / PROCESSOR                                 ####
###################################################################################################

corrector = ImageCorrector(IMG_DIR)
corrector.correct_images()
corrector.summary_final()

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

###################################################################################################
#                                          SAVE RESULTS                                        ####
###################################################################################################

fc.save("../Gis/Fishnet/fishnet_quarter_mile_completed.pkl")
