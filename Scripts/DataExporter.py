###################################################################################################
#                                              ATTRIBUTES                                      ####
###################################################################################################

TILE_SIZE_MILES = 0.25
BATCH_SIZE_MILES = 16  # 16
FISHNET_PATH = "./Gis/Fishnet/fishnet_quarter_mile_v2.pkl"
PERIODS = ["summer", "year"]
YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022]

SHAPEFILE_PATH = "./Gis/Texas_State_Boundary/State.shp"
# OR
COORDINATES = [-86.4466, 31.7655, -80.4437, 34.908]
FILTER = False
FILTER_REGION = [-86.4466, 31.7655, -80.4437, 34.908]

###################################################################################################
#                                              PACKAGES                                       ####
###################################################################################################

import sys
import warnings
import os

warnings.filterwarnings("ignore")

# import local modules
sys.path.append("./src")
sys.path.append("../src")
from Fishnet import Fishnet
from ImageExporter import ImageExporter

###################################################################################################
#                                         FISHNET GENERATION                                   ####
###################################################################################################

generate_fishnet = input("Do you want to generate a new Fishnet? (y/n)")
if generate_fishnet == "y":
    fc = Fishnet(
        tile_size_miles=TILE_SIZE_MILES,
        coordinates=COORDINATES,
        shapefile_path=None,
        clip=False,
        overlay_method=None,
    )
    fc.create_fishnet()
    fc.batch(BATCH_SIZE_MILES)
    if FILTER:
        fc.filter_fishnet_by_bbox(FILTER_REGION)
    fc.compute_neighbors()
else:
    print("Loading Fishnet... From file: ", FISHNET_PATH, "\n")
    fc = Fishnet.load(FISHNET_PATH)


###################################################################################################
#                                        GOOGLE EARTH ENGINE                                   ####
###################################################################################################

# Authenticate to Earth Engine
os.system("!earthengine authenticate")
os.system('ee.Initialize(project="jpmorgancapstone"')

###################################################################################################
#                                        GOOGLE EARTH ENGINE                                   ####
###################################################################################################


image_exporter = ImageExporter(fc, filtered=FILTER)

for pe in PERIODS:
    for year in YEARS:
        if pe == "summer":
            image_exporter.set_date_range(year, year, "05", "10")
        else:
            image_exporter.set_date_range(year, year, "01", "12")

        image_exporter.set_folder(f"{pe}_export_{year}")
        image_exporter.export_images()

print(
    "Done. Export should be running on the Google Earth Engine. See https://code.earthengine.google.com/ for details."
)
