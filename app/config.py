from sentinelhub import DataCollection
SUPPORTED_DATASET_EVALSCRIPTS = {
    "SENTINEL2-L1C": {
        "sentinel_sdk_obj": DataCollection.SENTINEL2_L1C,
        "supported_evalscripts": [
            {
                "name": "true-color",
                "path": "https://gist.githubusercontent.com/Rahul-7131/b02d5614401ba654904ff509039def15/raw/3867e78b12bf7d7dff44810c548ed20797b367ea/truecolor_wildfire.js",
                "scaling_factor": 1.5,
                "clip_range": {
                    "min": 0, 
                    "max": 1
                },
                "description": "A Sentinel-2 image highlighting areas of interest based on water, vegetation, and spectral thresholds in true color. Bands: B04, B03, B02, B08, B11, B12"
            }
        ]
    },
    "SENTINEL5P": {
        "sentinel_sdk_obj": DataCollection.SENTINEL5P,
        "supported_evalscripts": [
            {
                "name": "climate-bands",
                "path": "https://gist.githubusercontent.com/Rahul-7131/b02d5614401ba654904ff509039def15/raw/5c8894fe017e42c594a2fb755d10d57602049ec5/climate_evalscript.js",
                "scaling_factor": 1.5,
                "clip_range": {
                    "min": 0,
                    "max": 1
                },
                "description": "Carbon monoxide (CO) concentrations using a color ramp from low (blue) to high (red) and processes the image into a grid to determine dominant CO concentrations per grid cell."
            },
            {
                "name": "climate-mask",
                "path": "https://gist.githubusercontent.com/Rahul-7131/b02d5614401ba654904ff509039def15/raw/5c8894fe017e42c594a2fb755d10d57602049ec5/climate_evalscript.js",
                "scaling_factor": 255,
                "clip_range": {
                    "min": 0,
                    "max": 1
                },
                "description": "A mask of the carbon monoxide (CO) concentrations in the image. The mask is created by thresholding the CO concentrations in the image."
            },
            {
                "name": "fire-bands",
                "path": "https://gist.githubusercontent.com/Rahul-7131/b02d5614401ba654904ff509039def15/raw/3867e78b12bf7d7dff44810c548ed20797b367ea/wildfire_evalscript.js",
                "scaling_factor": 1.5,
                "clip_range": {
                    "min": 0,
                    "max": 1
                },
                "description": "Sentinel-2 image focussed on detection of wildfires, highlighting areas of interest based on vegetation (NDVI), water content (NDWI), and spectral thresholds in enhanced true color"
            },
            {
                "name": "fire-mask",
                "path": "https://gist.githubusercontent.com/Rahul-7131/b02d5614401ba654904ff509039def15/raw/5c8894fe017e42c594a2fb755d10d57602049ec5/climate_evalscript.js",
                "scaling_factor": 255,
                "clip_range": {
                    "min": 0,
                    "max": 1
                },
                "description": "A mask of the wildfire areas in the image. The mask is created by thresholding the NDVI and NDWI values in the image."
            },
        ]
    },
    # Add more datasets as needed
}
