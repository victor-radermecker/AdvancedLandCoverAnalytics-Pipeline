import ee


def dynamic_world(
    region=None,
    start_date="2020-01-01",
    end_date="2021-01-01",
    clip=False,
    reducer=None,
    projection="EPSG:3857",
    scale=10,
    return_type="hillshade",
    source = "GOOGLE/DYNAMICWORLD/V1",
):
    """Create 10-m land cover composite based on Dynamic World. The source code is adapted from the following tutorial by Spatial Thoughts:
    https://developers.google.com/earth-engine/tutorials/community/introduction-to-dynamic-world-pt-1

    Args:
        region (ee.Geometry | ee.FeatureCollection): The region of interest.
        start_date (str | ee.Date): The start date of the query. Default to "2020-01-01".
        end_date (str | ee.Date): The end date of the query. Default to "2021-01-01".
        clip (bool, optional): Whether to clip the image to the region. Default to False.
        reducer (ee.Reducer, optional): The reducer to be used. Default to None.
        projection (str, optional): The projection to be used for creating hillshade. Default to "EPSG:3857".
        scale (int, optional): The scale to be used for creating hillshade. Default to 10.
        return_type (str, optional): The type of image to be returned. Can be one of 'hillshade', 'visualize', 'class', or 'probability'. Default to "hillshade".

    Returns:
        ee.Image: The image with the specified return_type.
    """

    if return_type not in ["hillshade", "visualize", "class", "probability"]:
        raise ValueError(
            f"{return_type} must be one of 'hillshade', 'visualize', 'class', or 'probability'."
        )

    if reducer is None:
        reducer = ee.Reducer.mode()

    dw = ee.ImageCollection(source).filter(
        ee.Filter.date(start_date, end_date)
    )

    if isinstance(region, ee.FeatureCollection) or isinstance(region, ee.Geometry):
        dw = dw.filterBounds(region)
    else:
        raise ValueError("region must be an ee.FeatureCollection or ee.Geometry.")

    # Create a Mode Composite
    classification = dw.select("label")
    dwComposite = classification.reduce(reducer)
    if clip and (region is not None):
        if isinstance(region, ee.Geometry):
            dwComposite = dwComposite.clip(region)
        elif isinstance(region, ee.FeatureCollection):
            dwComposite = dwComposite.clipToCollection(region)
        elif isinstance(region, ee.Feature):
            dwComposite = dwComposite.clip(region.geometry())

    if source == "GOOGLE/DYNAMICWORLD/V1":
        dwVisParams = {
            "min": 0,
            "max": 8,
            "palette": [
                "#419BDF",
                "#397D49",
                "#88B053",
                "#7A87C6",
                "#E49635",
                "#DFC35A",
                "#C4281B",
                "#A59B8F",
                "#B39FE1",
            ],
        }
    elif source == "USGS/NLCD_RELEASES/2019_REL/NLCD":
        dwVisParams = {
            "min": 0,
            "max": 20,
            "palette": [
                "#466b9f", # Open Water (11)
                "#d1def8", # Perennial Ice/Snow (12)
                "#dec5c5", # Developed, Open Space (21)
                "#d99282", # Developed, Low Intensity (22)
                "#eb0000", # Developed, Medium Intensity (23)
                "#ab0000", # Developed, High Intensity (24)
                "#b3ac9f", # Barren Land (31)
                "#68ab5f", # Deciduous Forest (41)
                "#1c5f2c", # Evergreen Forest (42)
                "#b5c58f", # Mixed Forest (43)
                "#af963c", # Dwarf Scrub (51)
                "#ccb879", # Shrub/Scrub (52)
                "#dfdfc2", # Grassland/Herbaceous (71)
                "#d1d182", # Sedge/Herbaceous (72)
                "#a3cc51", # Lichens (73)
                "#82ba9e", # Moss (74)
                "#dcd939", # Pasture/Hay (81)
                "#ab6c28", # Cultivated Crops (82)
                "#b8d9eb", # Woody Wetlands (90)
                "#6c9fb8", # Emergent Herbaceous Wetlands (95)
            ],
        }
    else:
        raise ValueError(
            f"{source} is not supported. Please use 'GOOGLE/DYNAMICWORLD/V1' or 'USGS/NLCD_RELEASES/2019_REL/NLCD'."
        )

    if return_type == "class":
        return dwComposite
    elif return_type == "visualize":
        return dwComposite.visualize(**dwVisParams)
    else:
        # Create a Top-1 Probability Hillshade Visualization
        probabilityBands = [
            "water",
            "trees",
            "grass",
            "flooded_vegetation",
            "crops",
            "shrub_and_scrub",
            "built",
            "bare",
            "snow_and_ice",
        ]

        # Select probability bands
        probabilityCol = dw.select(probabilityBands)

        # Create a multi-band image with the average pixel-wise probability
        # for each band across the time-period
        meanProbability = probabilityCol.reduce(ee.Reducer.mean())

        # Composites have a default projection that is not suitable
        # for hillshade computation.
        # Set a EPSG:3857 projection with 10m scale
        proj = ee.Projection(projection).atScale(scale)
        meanProbability = meanProbability.setDefaultProjection(proj)

        # Create the Top1 Probability Hillshade
        top1Probability = meanProbability.reduce(ee.Reducer.max())

        if clip and (region is not None):
            if isinstance(region, ee.Geometry):
                top1Probability = top1Probability.clip(region)
            elif isinstance(region, ee.FeatureCollection):
                top1Probability = top1Probability.clipToCollection(region)
            elif isinstance(region, ee.Feature):
                top1Probability = top1Probability.clip(region.geometry())

        if return_type == "probability":
            return top1Probability
        else:
            top1Confidence = top1Probability.multiply(100).int()
            hillshade = ee.Terrain.hillshade(top1Confidence).divide(255)
            rgbImage = dwComposite.visualize(**dwVisParams).divide(255)
            probabilityHillshade = rgbImage.multiply(hillshade)

            return probabilityHillshade