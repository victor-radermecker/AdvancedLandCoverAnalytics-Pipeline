###################################################################################################
#                                              ATTRIBUTES                                      ####
###################################################################################################

TILE_SIZE_MILES = 0.25
BATCH_SIZE_MILES = 16  # 16
FILTER_REGION = [
    -99.13,
    28.91,
    -94.29,
    31.1,
]  # [-95.799944, 29.374853, -95.028636, 29.795492,]
IMG_DIR = "./Images3/"
SAVE_DIR = "./Outputs/"
SHAPEFILE_PATH = "./Gis/Texas_State_Boundary/State.shp"
FISHNET_PATH = "./Gis/Fishnet/fishnet_quarter_mile_v2.pkl"


###################################################################################################
#                                               PACKAGES                                       ####
###################################################################################################

# import modules
import sys
import warnings
import os
from datetime import datetime
from tqdm import tqdm

warnings.filterwarnings("ignore")

# import local modules
sys.path.append("./src")
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
export_img = input(
    "Are all images exported from Gogole Earth and stored locally? (y/n)"
)
if export_img == "n":
    raise Exception("Please export images from Google Earth Engine first.")

generate_fishnet = input("Do you want to generate a new Fishnet? (y/n)")
if generate_fishnet == "y":
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
else:
    print("Loading Fishnet... From file: ", FISHNET_PATH, "\n")
    fc = Fishnet.load(FISHNET_PATH)


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

timestamp = datetime.today().strftime("%Y-%m-%d")
print("Data Processing Finished. Saving results...")
fc.save(f"{SAVE_DIR}fishnet_quarter_mile_completed_{timestamp}.pkl")
