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
            },
            {
                "name": "natural",
                "path": "https://gist.githubusercontent.com/Rahul-7131/3a500efecf5dbd5af7ebe7cdcc87a0e9/raw/451e3fac3efb0cdfda992f9097940d9e43a29028/natural.js",
                "scaling_factor": 1.5,
                "clip_range": {
                    "min": 0, 
                    "max": 1
                },
                "description" : "A Sentinel-2 image highlighting areas of interest based on water, vegetation, and spectral thresholds in natural color. Bands: B04, B03, B02, B08, B11, B12"
            }
        ]
    },
    "SENTINEL2-L2A": {
        "sentinel_sdk_obj": DataCollection.SENTINEL2_L2A,
        "supported_evalscripts": [
            {
                "name" : "moisture",
                "path" : "https://gist.githubusercontent.com/Rahul-7131/3a500efecf5dbd5af7ebe7cdcc87a0e9/raw/451e3fac3efb0cdfda992f9097940d9e43a29028/moisture.js",
                "clip_range": {
                    "min": 0,
                    "max": 1
                },
                "description" : "A Sentinel-2 image highlighting areas of interest based on water content in the image.",
            
            }            
        ]
    },
    "SENTINEL3-OLCI": {
        "sentinel_sdk_obj": DataCollection.SENTINEL3_OLCI,
        "supported_evalscripts": [
            {
                "name" : "chlorophyll",
                "path" : "https://gist.githubusercontent.com/Rahul-7131/3a500efecf5dbd5af7ebe7cdcc87a0e9/raw/451e3fac3efb0cdfda992f9097940d9e43a29028/chlorophyll.js",
                "clip_range": {
                    "min": 0,
                    "max": 1
                },
                "description" : "A Sentinel-3 image highlighting areas of interest based on chlorophyll content in the image.",
            }
        ]
    },
    "SENTINEL3-SLSTR": {
        "sentinel_sdk_obj": DataCollection.SENTINEL3_SLSTR,
        "supported_evalscripts": [
            {
                "name" : "thermal",
                "path" : "https://gist.githubusercontent.com/Rahul-7131/3a500efecf5dbd5af7ebe7cdcc87a0e9/raw/451e3fac3efb0cdfda992f9097940d9e43a29028/thermal.js",
                "clip_range": {
                    "min": 0,
                    "max": 1
                },
                "description" : "A Sentinel-3 image highlighting areas of interest based on thermal content in the image.",
            }
        ]    
    },
    "SENTINEL5P": {
        "sentinel_sdk_obj": DataCollection.SENTINEL5P,
        "supported_evalscripts": [
            {
                "name": "aerosol",
                "path": "https://gist.githubusercontent.com/Rahul-7131/3a500efecf5dbd5af7ebe7cdcc87a0e9/raw/451e3fac3efb0cdfda992f9097940d9e43a29028/aerosol.js",
                "scaling_factor": 1.5,
                "clip_range": {
                    "min": 0,
                    "max": 1
                },
                "description" : "Aerosol optical depth (AOD) using a color ramp."
            },
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
            {
                "name" : "optical-thickness",
                "path" : "https://gist.githubusercontent.com/Rahul-7131/3a500efecf5dbd5af7ebe7cdcc87a0e9/raw/451e3fac3efb0cdfda992f9097940d9e43a29028/optical_thickness.js",
                "scaling_factor": 1.5,
                "clip_range": {
                    "min": 0,
                    "max": 1
                },
                "description" : "A Sentinel-5P image highlighting areas of interest based on cloud cover in the image."
            }
        ]
    },
    # Add more datasets as needed
}
