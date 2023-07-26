# 🌆 Capstone: JPMorgan Urbanization Analysis

**Project Objective:** Utilize satellite & GIS data, along with socio-demographic indicators, to quantify urbanization rates. This analysis will inform JPMorgan's client base, support existing hubs, and identify new opportunities.

## 🚀 Getting Started

Follow these instructions to set up the project on your local machine.

### 1. Clone the Repository

```bash
git clone git@github.com:victor-radermecker/Capstone_JPMorgan.git
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
