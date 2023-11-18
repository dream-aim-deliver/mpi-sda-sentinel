import os
import json
import pandas as pd
from dotenv import load_dotenv
from sentinelhub import SHConfig, BBox, CRS, DataCollection, SentinelHubRequest, bbox_to_dimensions, MimeType
from PIL import Image
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut
from utils import plot_image
from models import PipelineRequestModel
import logging

# Load environment variables
load_dotenv()

class SentinelHubPipelineElement:
    """
    A class to handle the pipeline for fetching, processing, and saving satellite images
    from Sentinel Hub.

    Attributes:
        lfn (str): Local filename for the data file.
        start_date (str): Start date for satellite image retrieval.
        end_date (str): End date for satellite image retrieval.
        image_dir (str): Directory path for saving the images.
    """

    def __init__(self, request: PipelineRequestModel, start_date: str, end_date: str, image_dir: str = None, resolution: int = 60) -> None:
        """
        Initializes the SentinelHubPipelineElement with the given request and date range.

        Args:
            request (PipelineRequestModel): A request model object containing the necessary parameters.
            start_date (str): The start date for the data request in 'YYYY-MM-DD' format.
            end_date (str): The end date for the data request in 'YYYY-MM-DD' format.
            image_dir (str, optional): Directory path for saving images. Defaults to the path from environment variable IMAGE_SAVE_PATH.
        """
        self.lfn = request.lfn
        self.start_date = start_date
        self.end_date = end_date
        self.image_dir = image_dir or os.getenv("IMAGE_SAVE_PATH")
        self.resolution = resolution

    def get_sentinel_hub_config(self):
        """
        Configures and saves the Sentinel Hub configuration using client ID and secret from environment variables.
        """
        CLIENT_ID = os.getenv("CLIENT_ID")
        CLIENT_SECRET = os.getenv("CLIENT_SECRET")

        self.config = SHConfig()
        if CLIENT_ID and CLIENT_SECRET:
            self.config.sh_client_id = CLIENT_ID
            self.config.sh_client_secret = CLIENT_SECRET
        else:
            logging.warning("CLIENT_ID or CLIENT_SECRET not found in environment variables.")
        
        self.config.save()

    def load_data(self, lfn: str) -> pd.DataFrame:
        """
        Loads data from a JSON file.

        Args:
            lfn (str): The local filename of the JSON file.

        Returns:
            pd.DataFrame: A DataFrame containing the loaded data.
        """
        try:
            with open(lfn, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return pd.DataFrame(data)
        except FileNotFoundError:
            logging.error(f"File not found: {lfn}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON file: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error while loading data: {e}")
            raise

    def generate_bounding_box(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates bounding boxes for each location in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing location data.

        Returns:
            pd.DataFrame: Updated DataFrame with bounding boxes.
        """
        if 'extracted.location' not in df.columns:
            logging.error("DataFrame does not contain 'extracted.location' column.")
            raise ValueError("Missing 'extracted.location' column in DataFrame.")

        df['bounding_box'] = df['extracted.location'].apply(self.get_bounding_boxes)
        return df

    @staticmethod
    def get_bounding_boxes(place_name):
        """
        Fetches bounding box for a given place name.

        Args:
            place_name (str): The name of the place to find the bounding box for.

        Returns:
            dict: A dictionary containing the bounding box, or None if place is not found.
        """
        try:
            geolocator = Nominatim(user_agent="place_boundary")
            location = geolocator.geocode(place_name)
            if location:
                lat, lon = location.latitude, location.longitude
                bounding_box = {'min_latitude': lat - 0.16, 'max_latitude': lat + 0.16,
                                'min_longitude': lon - 0.35, 'max_longitude': lon + 0.35}
                return bounding_box
            else:
                return None
        except GeocoderUnavailable:
            logging.error("Geocoder service is unavailable.")
            raise
        except GeocoderTimedOut:
            logging.error("Geocoder service timed out.")
            raise
        except Exception as e:
            logging.error(f"Unexpected error in geocoding: {e}")
            raise

    def generate_coordinate_lists(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates lists of coordinates and their sizes for the bounding boxes in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing bounding box data.

        Returns:
            pd.DataFrame: DataFrame containing coordinates and their sizes.
        """
        coords_list, coords_size_list = [], []
        for _, row in df.iterrows():
            bounding_box = row.get('bounding_box')
            if bounding_box:
                min_lat, max_lat, min_lon, max_lon = bounding_box.values()
                coords_bbox = BBox((min_lon, min_lat, max_lon, max_lat), crs=CRS.WGS84)
                coords_size = bbox_to_dimensions(coords_bbox, resolution=self.resolution)
                coords_list.append((min_lon, min_lat, max_lon, max_lat))
                coords_size_list.append(coords_size)
            else:
                coords_list.append(())
                coords_size_list.append(None)
        return pd.DataFrame({'coords': coords_list, 'coords_size': coords_size_list}).dropna()

    def execute(self) -> None:
        """
        Main execution function that runs the pipeline process.
        """
        df = self.load_data(self.lfn)
        df = self.generate_bounding_box(df)
        df_coords = self.generate_coordinate_lists(df)
        config = self.get_sentinel_hub_config()
        evalscript_true_color = self.get_satellite_bands_config()
        true_color_images = self.get_images(df_coords, evalscript_true_color, config, self.start_date, self.end_date)
        self.plot_and_save_images(true_color_images)

    @staticmethod
    def get_satellite_bands_config() -> str:
        """
        Returns the evalscript configuration for Sentinel Hub.

        Returns:
            str: Evalscript for true color imagery.
        """
        return """
            //VERSION=3
            function setup() {
                return {
                    input: [{bands: ["B02", "B03", "B04"]}],
                    output: {bands: 3}
                };
            }
            function evaluatePixel(sample) {
                return [sample.B04, sample.B03, sample.B02];
            }
        """

    def get_images(self, df_coords: pd.DataFrame, evalscript_true_color: str, config: SHConfig, start_date: str, end_date: str):
        """
        Retrieves images for each set of coordinates in the DataFrame within the specified date range.

        Args:
            df_coords (pd.DataFrame): DataFrame containing coordinates and their sizes.
            evalscript_true_color (str): Evalscript for Sentinel Hub request.
            config (SHConfig): Sentinel Hub configuration object.
            start_date (str): Start date for image retrieval.
            end_date (str): End date for image retrieval.

        Returns:
            list: List of retrieved images.
        """
        images = []
        for _, row in df_coords.iterrows():
            coords, coords_size = row['coords'], row['coords_size']
            if coords:
                min_lon, min_lat, max_lon, max_lat = coords
                coords_bbox = BBox((min_lon, min_lat, max_lon, max_lat), crs=CRS.WGS84)
                request_true_color = SentinelHubRequest(
                    evalscript=evalscript_true_color,
                    input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L1C, time_interval=(start_date, end_date))],
                    responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
                    bbox=coords_bbox, size=coords_size, config=config
                )
                images.extend(request_true_color.get_data())
        return images

    def plot_and_save_images(self, true_color_imgs: list) -> None:
        """
        Plots and saves the retrieved images.

        Args:
            true_color_imgs (list): A list of images to be plotted and saved.
        """
        for index, image in enumerate(true_color_imgs):
            plot_image(image, factor=3.5 / 255, clip_range=(0, 1))
            image_filename = f"true_color_image_{index}.png"
            image_path = os.path.join(self.image_dir, image_filename)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            Image.fromarray((image * 255).astype('uint8')).save(image_path)
            print(f"Image saved to: {image_path}")