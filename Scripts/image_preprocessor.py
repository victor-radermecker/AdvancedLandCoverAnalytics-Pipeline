###################################################################################################
#                                              ATTRIBUTES                                      ####
###################################################################################################

TILE_SIZE_MILES = 0.25
BATCH_SIZE_MILES = 16  # 16
IMG_DIR = "./Images/Train"  # ./Archives/Images/

SAVE_DIR = "./Outputs/"
FILE_NAME = "urbanization_train"

FISHNET_PATH = "./Gis/Fishnet/fishnet_quarter_mile_v2.pkl"
# OR
SHAPEFILE_PATH = "./Gis/Texas_State_Boundary/State.shp"
# OR
COORDINATES = [-99.13, 28.91, -94.29, 31.1]
FILTER = True
FILTER_REGION = [-99.13, 28.91, -94.29, 31.1]

###################################################################################################
#                                               PACKAGES                                       ####
###################################################################################################

# import modules
import sys
import warnings
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm

warnings.filterwarnings("ignore")

# import local modules
sys.path.append("./src")
from Fishnet import Fishnet
from ImageCorrector import ImageCorrector
from ImageProcessor import ImageProcessor

# check the directories exist
if not os.path.exists(IMG_DIR):
    print(f"Creating directory {IMG_DIR}.")
    os.makedirs(IMG_DIR)
if not os.path.exists(SAVE_DIR):
    print(f"Creating directory {SAVE_DIR}.")
    os.makedirs(SAVE_DIR)

csv = input("Do you want to save the final .CSV and Metadata files? (y/n)")

###################################################################################################
#                                         FISHNET GENERATION                                   ####
###################################################################################################

# Generate a fishnet using a shapefile
export_img = input(
    "Are all images exported from Google Earth and stored locally? (y/n)"
)
if export_img == "n":
    raise Exception("Please export images from Google Earth Engine first.")

generate_fishnet = input("Do you want to generate a new Fishnet? (y/n)")
if generate_fishnet == "y":
    dec = input(
        "Do you want to use a ShapeFile or Coordinates to generate the Fishnet? (s/c) "
    )
    if dec == "s":
        fc = Fishnet(
            tile_size_miles=TILE_SIZE_MILES,
            coordinates=None,
            shapefile_path=SHAPEFILE_PATH,
            clip=False,
            overlay_method=None,
        )
    elif dec == "c":
        fc = Fishnet(
            tile_size_miles=TILE_SIZE_MILES,
            coordinates=COORDINATES,
            shapefile_path=None,
            clip=False,
            overlay_method=None,
        )
    else:
        raise Exception("Please enter a valid option.")
    fc.create_fishnet()
    fc.batch(BATCH_SIZE_MILES)
    if FILTER:
        fc.filter_fishnet_by_bbox(FILTER_REGION)
    fc.compute_neighbors()
else:
    print("Loading Fishnet... From file: ", FISHNET_PATH, "\n")
    fc = Fishnet.load(FISHNET_PATH)


###################################################################################################
#                                  IMAGE CORRECTOR / PROCESSOR                                 ####
###################################################################################################

# corrector = ImageCorrector(IMG_DIR, verbose=False)
# corrector.correct_images()

img_process = ImageProcessor(
    fc,
    filtered=FILTER,
)
img_process.assign_fishnet_tiles_to_pixels(
    IMG_DIR + f"/{2016}/Final/", "landcover_batchID"
)
for year in tqdm([2016, 2017, 2018, 2019, 2020, 2021, 2022]):
    # Compute urbanization rate
    img_process.compute_mean_tile_entropy_urbanization(
        IMG_DIR + f"/{year}/Final/",
        "landcover_batchID",
        f"MeanPixel_{year}",
        f"Entropy_{year}",
    )

if csv:
    img_process.generate_processed_csv(SAVE_DIR, FILE_NAME, FILTER)


###################################################################################################
#                                          SAVE RESULTS                                        ####
###################################################################################################

timestamp = datetime.today().strftime("%Y-%m-%d")
print("Data Processing Finished. Saving results...")
fc.save(f"{SAVE_DIR}fishnet_quarter_mile_completed_{timestamp}.pkl")
