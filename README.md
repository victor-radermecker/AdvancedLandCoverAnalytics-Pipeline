# 🌆 Enabling Advanced Land Cover Analytics: An Integrated Data Extraction Pipeline for Predictive Modeling with the Dynamic World Dataset

## Introduction
This repository presents a flexible and efficient end-to-end pipeline for extracting, preprocessing, and representing data from the Dynamic World dataset. This dataset, a near-real-time land use/land cover (LULC) dataset, facilitates advanced land cover analytics and predictive modeling. The pipeline was developed to simplify the data extraction process, reduce the learning curve associated with remote sensing data, and provide standardized data preprocessing for various downstream tasks, including urbanization prediction.

### How the Pipeline Works
The pipeline is designed to handle the significant volume and potential noise inherent in satellite data. Here is a detailed explanation of the components and workflow:

1. Fishnet Generation: The region of interest is divided into fixed-sized tiles (fishnet). This allows detailed analysis at a granular level, facilitating easier data management and comparative analysis across regions. The fishnet grid is generated using latitude and longitude boundaries or a provided shapefile, specifying the size of each tile.

2. Image Extraction: The ImageExporter module uses the Earth Engine Python API, building upon the GeeMap package, to convert Python code into JavaScript queries. These queries are dispatched to the Earth Engine API to extract images, which are stored in a designated Google Drive folder.

3. Image Aggregation and Correction: To obtain a single image per year for each tile, multiple images are extracted and aggregated to reduce noise. Aggregation techniques include averaging, minimizing, and maximizing pixel values. The ImageCorrector module handles the aggregation and applies imputation techniques to fill gaps caused by cloud cover or other missing data.

4. Seasonal Variation Handling: Images are extracted from summer months (June 1st to October 1st) to avoid snow coverage and better capture vegetation and land cover features. This approach reduces seasonal noise and ensures consistent data quality.

5. Data Processing and Feature Extraction: The ImageProcessor class computes various aggregate metrics, such as the percentage of pixels with specific values within each tile. These metrics are stored in a tabular format, indexed by the unique fishnet tile identifier.

6. Integration with Machine Learning Models: The processed data, including images and tabular metrics, are seamlessly integrated into machine learning models. This allows for efficient analysis and prediction of land cover changes, such as urbanization.

### Case Study: Urbanization Prediction Using Dynamic World Data

To validate our pipeline, we applied it to predict urbanization trends in a rapidly developing region. We partitioned the area into a grid and extracted annual composite images from 2016 to 2022 using the Dynamic World dataset. These images were processed to reduce noise and fill gaps, providing clean data for analysis. Our hybrid model, combining XGBoost and ConvLSTM (XGCLM), classified regions by urbanization activity and predicted future growth. The pipeline demonstrated high prediction accuracy, particularly in rapidly urbanizing areas, underscoring its effectiveness for large-scale land cover analysis and supporting sustainable urban planning.

## 🚀 Getting Started

Follow these instructions to set up the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/victor-radermecker/Urbanization-Rate-Analysis-Through-Dynamic-World-Based-Video-Prediction.git
```

Create a conda environment using the `requirements.yml` file.

```bash
conda env create -f requirements.yml
```

Activate the conda environment using:

```bash
conda activate capstone
```

### 2. Download the Data

All files are available here:

[Dropbox Link](https://www.dropbox.com/scl/fo/i6r9qx73a0lervrd2crpk/h?dl=0&rlkey=g8twup5jtib6h3xnle353dvtg)

The structure of the project folder should be as follows:

```
Capstone_JPMorgan
├── Images
│   ├── Train
│   │   ├── 2016
│   │   │   ├── Summer
│   │   │   │   ├── ...
│   │   │   ├── Year
│   │   │   │   ├── ...
│   │   │   ├── Final
│   │   │   │   ├── ...
│   │   ├── 2016
│   │   │   ├── ...
│   │   ├── ...
│   │   ├── 2022
│   │   │   ├── ...
│   ├── Valid
│   │   ├── ...
│   ├── Test
│   │   ├── ...
│   ├── ...
├── Gis
│   ├── ...
├── src
│   ├── ...
```

### 3. Run the Notebooks

### 4. Refresh the YML conda environment file

Use the following command to refresh the requirements.yml file used to generate the conda environment.

```bash
conda env export --file environment.yml
```
