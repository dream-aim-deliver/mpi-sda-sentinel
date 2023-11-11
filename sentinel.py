import pandas as pd
from dateutil import parser
from pydantic import BaseModel
import spacy
import re
from geopy.geocoders import Nominatim
import geopandas as gpd
from shapely.geometry import Point
import json
import logging
from sentinelhub import SHConfig   
import datetime
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
from geopy.exc import GeocoderUnavailable

from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
)
from models import PipelineRequestModel, SentinalHubRequest
from utils import plot_image
from dotenv import load_dotenv

 # Set up logging
    # TODO - Use root logger instead of creating a new one

logger = logging.getLogger(__name__)

class SentinalHubPipelineElement:
    def __init__(self, request: PipelineRequestModel) -> None:
        self.logger = logger
        load_dotenv()
        self.lfn = request.lfn

    def get_sentinal_hub_config(self):
        CLIENT_ID = os.getenv("CLIENT_ID")
        CLIENT_SECRET = os.getenv("CLIENT_SECRET")
        self.config = SHConfig()
        self.config.sh_client_id = CLIENT_ID
        self.config.sh_client_secret = CLIENT_SECRET
        self.config.save()

    # Load data from JSON file
    def load_data(self, lfn: str) -> pd.DataFrame:
        # TODO: Download file from MinIO instead of reading from local file
        with open(lfn, 'r', encoding='utf-8') as file:
            data = json.load(file)
            df = pd.DataFrame(data)
            # TODO do we need to store in a more permanent context?
            return df
    
    def generate_bounding_box(self, df: pd.DataFrame) -> pd.DataFrame:
        df['bounding_box'] = df['extracted.location'].apply(self.get_bounding_boxes)
        return df
    

    def generate_coordinate_lists(self, df: pd.DataFrame) -> pd.DataFrame:
        resolution = 60
        coords_list = []  # List to store the calculated coordinates
        coords_size_list = []  # List to store the calculated sizes

        for index, row in df.iterrows():
            bounding_box_dict = row['bounding_box']

            if bounding_box_dict is not None:
                min_lat = bounding_box_dict['min_latitude']
                max_lat = bounding_box_dict['max_latitude']
                min_lon = bounding_box_dict['min_longitude']
                max_lon = bounding_box_dict['max_longitude']

                coords_wgs84 = (min_lon, min_lat, max_lon, max_lat)
                coords_bbox = BBox(bbox=coords_wgs84, crs=CRS.WGS84)
                coords_size = bbox_to_dimensions(coords_bbox, resolution=resolution)

                # Append the values directly without naming them
                coords_list.append((min_lon, min_lat, max_lon, max_lat))
                coords_size_list.append(coords_size)
            else:
                # Append an empty tuple and None for coords_size if 'bounding_box' is None
                coords_list.append(())
                coords_size_list.append(None)

        # Create a new DataFrame with 'coords' and 'coords_size' columns
        df2 = pd.DataFrame({
            'coords': coords_list,
            'coords_size': coords_size_list
        })
        df2.dropna(inplace=True)
        return df2

    def execute(self) -> None:
        df = self.load_data(self.lfn)
        df = self.get_bounding_boxes(df)
        df_coords = self.generate_coordinate_lists(df)
        config = self.get_sentinal_hub_config()
        evalscript_true_color = self.get_satellite_bands_config()
        start_date = "2023-01-09"
        end_date = "2023-10-09"
        true_color_images = self.get_images(df_coords, evalscript_true_color, config, start_date, end_date)
        self.plot_and_save_images(true_color_images)

    def get_bounding_boxes(place_name):
        geolocator = Nominatim(user_agent="place_boundary")
        location = geolocator.geocode(place_name)
        
        if location:
            # Get latitude and longitude
            lat, lon = location.latitude, location.longitude
            
            # Define a bounding box around the location (adjust the size as needed)
            bounding_box = {
                'min_latitude': lat - 0.16,
                'max_latitude': lat + 0.16,
                'min_longitude': lon - 0.35,
                'max_longitude': lon + 0.35
            }
            
            return bounding_box
        else:
            return None

    def get_satellite_bands_config():
    # SentinelHub evalscript for true color imagery
        evalscript_true_color = """
            //VERSION=3

            function setup() {
                return {
                    input: [{
                        bands: ["B02", "B03", "B04"]
                    }],
                    output: {
                        bands: 3
                    }
                };
            }

            function evaluatePixel(sample) {
                return [sample.B04, sample.B03, sample.B02];
            }
        """
        return evalscript_true_color

    # Loop through DataFrame and request true color images
    # TODO: Must be processed in parallel
    def get_images(df_coords: pd.DataFrame, evalscript_true_color: str, config: SHConfig, start_date: str, end_date: str):
        for index, row in df_coords.iterrows():
            coords = row['coords']
            coords_size = row['coords_size']

            if coords:
                min_lon, min_lat, max_lon, max_lat = coords
                coords_bbox = BBox((min_lon, min_lat, max_lon, max_lat), crs=CRS.WGS84)

                # Request true color images
                request_true_color = SentinelHubRequest(
                    evalscript=evalscript_true_color,
                    input_data=[
                        SentinelHubRequest.input_data(
                            data_collection=DataCollection.SENTINEL2_L1C,
                            # TODO - Use the date from the JSON file instead of hardcoding
                            time_interval=("2023-01-09", "2023-10-09"),
                        )
                    ],
                    responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
                    bbox=coords_bbox,
                    size=coords_size,
                    config=config,
                )
                true_color_imgs = request_true_color.get_data()
        return true_color_imgs
    def plot_and_save_images(true_color_imgs: list) -> None:
        for index, image in enumerate(true_color_imgs):
            # Plot and save images
            image = true_color_imgs[0]
            plot_image(image, factor=3.5 / 255, clip_range=(0, 1))

            image_filename = f"true_color_image_{index}.png"
            image_path = os.path.join("C:/Users/Mihir Trivedi/Desktop/final code/img_generated", image_filename)

            # TODO should already exist on application startup
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            im = Image.fromarray((image * 255).astype('uint8'))
            im.save(image_path)

            print(f"Image saved to: {image_path}")

        # Additional code for handling the returned data
        print(f"Returned data is of type = {type(true_color_imgs)} and length {len(true_color_imgs)}.")
        print(f"Single element in the list is of type {type(true_color_imgs[-1])} and has shape {true_color_imgs[-1].shape}")
