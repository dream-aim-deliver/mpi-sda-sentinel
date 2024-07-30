from logging import Logger
import logging
from typing import List
from sentinelhub import SHConfig, BBox, CRS, DataCollection, SentinelHubRequest, bbox_to_dimensions, MimeType
from app.sdk.models import KernelPlancksterSourceData, BaseJobState, JobOutput, ProtocolEnum
from app.sdk.scraped_data_repository import ScrapedDataRepository,  KernelPlancksterSourceData
from augmentations.climate_augmentations import augment_climate_images
from augmentations.wildfire_augmentations import augment_wildfire_images
import time
import os
import json
import pandas as pd
from PIL import Image
from utils import date_range, save_image
from models import PipelineRequestModel
from numpy import ndarray
import numpy as np
import cv2
import shutil

def load_evalscript(filepath: str) -> str:
    with open(filepath, 'r') as file:
        return file.read()

def get_true_wildfire_config():
     return """
                function setup() {
                return {
                    input: ["B02", "B03", "B04", "B08", "B11", "B12", "dataMask"],
                    output: { bands: 4 }
                    };
                }

                function evaluatePixel(samples) {
                    var NDWI=index(samples.B03, samples.B08); 
                    var NDVI=index(samples.B08, samples.B04);
                    var INDEX= ((samples.B11 - samples.B12) / (samples.B11 + samples.B12))+(samples.B08);

                    if((INDEX>0.1)||(samples.B02>0.1)||(samples.B11<0.1)||(NDVI>0.3)||(NDWI > 0.1)){
                        return[2.5*samples.B04, 2.5*samples.B03, 2.5*samples.B02, samples.dataMask]
                    }
                    else {
                    return [1, 0, 0, samples.dataMask]
                    }
                }
            """
def get_true_climate_config():
    return """
        function setup() {
            return {
                input: ["CO", "dataMask"], 
                output: { bands: 4 }
            };
        }

        function evaluatePixel(samples) {
            var CO_threshold = 0.15; // Define a threshold for high CO levels

            // might need normalization
            var CO_normalized = samples.CO / 0.15; // max CO value expected is 0.15

            if (CO_normalized > 1) CO_normalized = 1; // Capped the value at 1 for safety
            if (CO_normalized > 0.5) { // Above 50 of the threshold
                return [1, 0, 0, samples.dataMask]; // Highlight in red for high CO
            } else if (CO_normalized > 0.2) { // Between 20 and 50 of the threshold
                return [1, 1, 0, samples.dataMask]; // Highlight in yellow for moderate CO
            } else {
                return [0, 1, 0, samples.dataMask]; // Highlight in green for low CO
            }
        }
    """

def get_images(logger: Logger, job_id: int, tracer_id: str, scraped_data_repository: ScrapedDataRepository, 
               output_data_list: list[KernelPlancksterSourceData], protocol: ProtocolEnum, 
               coords_wgs84: tuple[float, float, float, float], evalscript_bands_config: str, config: SHConfig, start_date: str, end_date: str, 
               resolution: int, image_dir: str, augmentation_type: str):
    """
    Retrieves images for each set of coordinates within the specified date range.

    Args:
        logger (Logger): Logger for logging information.
        job_id (int): Job identifier.
        tracer_id (str): Tracer identifier.
        scraped_data_repository (ScrapedDataRepository): Repository for scraped data.
        output_data_list (list): List to store output data.
        protocol (ProtocolEnum): Protocol type for data.
        coords_wgs84 (tuple): Coordinates in WGS84 format.
        evalscript_bands_config (str): Evalscript for specific band configuration.
        evalscript_truecolor (str): Evalscript for true color imagery.
        config (SHConfig): Sentinel Hub configuration object.
        start_date (str): Start date for image retrieval.
        end_date (str): End date for image retrieval.
        resolution (int): Resolution for image retrieval.
        image_dir (str): Directory to save images.
        augmentation_type (str): Type of augmentation applied.

    Returns:
        list: List of retrieved and processed images.
    """
    images = []
    coords_bbox = BBox(bbox=coords_wgs84, crs=CRS.WGS84)
    coords_size = bbox_to_dimensions(coords_bbox, resolution=resolution)
    date_intervals = date_range(start_date, end_date)
    if augmentation_type == "wildfire":
        evalscript_truecolor = get_true_wildfire_config()
    elif augmentation_type == "climate":
        evalscript_truecolor = get_true_climate_config()
    #logging.log(f"Image shape at {resolution} m resolution: {coords_size} pixels")
    dataset = None
    if augmentation_type == "wildfire":
        dataset = DataCollection.SENTINEL2_L1C
    elif augmentation_type == "climate":
        dataset = DataCollection.SENTINEL5P

    

    for interval in date_intervals:
      
        try:
            request_bands_config = SentinelHubRequest(
                evalscript=evalscript_bands_config,
                input_data=[SentinelHubRequest.input_data(data_collection=dataset, time_interval=interval)],
                responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
                bbox=coords_bbox, size=coords_size, config=config
            )

        except Exception as e:
            logger.warn(e)
        
        data = None
        try:
            data = request_bands_config.get_data()
        except Exception as e:
            logger.warning(e)

        image = None
        if len(data)> 0:
            image = data[0]
 
        if np.mean(image) != 0.0: #if image is not entirely blank
                        
            #TODO: implment tempfile
            image_filename = f"{interval}_{augmentation_type}_banded_config.png"
            image_path = os.path.join(image_dir, "banded_config", image_filename)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            save_image(image, image_path, factor=1.5/255, clip_range=(0, 1))
            logger.info(f"Configured Bands Image saved to: {image_path}")

            
            data_name = str(interval).strip("()").replace("-","_").replace(",","_").replace("\'","").replace(" ","_") + "_" + augmentation_type + "_banded_config"
            relative_path = f"sentinel/{tracer_id}/{job_id}/banded_config/{data_name}.png"

        
            media_data = KernelPlancksterSourceData(
                name=data_name,
                protocol=protocol,
                relative_path=relative_path,
            )

            try:
                scraped_data_repository.register_scraped_photo(
                    job_id=job_id,
                    source_data=media_data,
                    local_file_name=image_path,
                )
            except Exception as e:
                logger.info("could not register file")

            output_data_list.append(media_data)
            #job.touch()
            

           

        

            image_filename = f"{interval}_{augmentation_type}_masked.png"
            image_path = os.path.join(image_dir, "masked", image_filename)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            save_image(image, image_path, factor=255/255, clip_range=(0, 1))
            logger.info(f"Masked Image saved to: {image_path}")

            data_name = str(interval).strip("()").replace("-","_").replace(",","_").replace("\'","").replace(" ","_") + "_" + augmentation_type + "_masked"
            
            relative_path = f"sentinel/{tracer_id}/{job_id}/masked/{data_name}.png"

        
            media_data = KernelPlancksterSourceData(
                name=data_name,
                protocol=protocol,
                relative_path=relative_path,
            )

            try:
                scraped_data_repository.register_scraped_photo(
                    job_id=job_id,
                    source_data=media_data,
                    local_file_name=image_path,
                )
            except Exception as e:
                logger.info("could not register file")

            output_data_list.append(media_data)
           
    for interval in date_intervals:
      
        try:
        
            request_truecolor = SentinelHubRequest(
                evalscript=evalscript_truecolor,
                input_data=[SentinelHubRequest.input_data(data_collection= dataset, time_interval=interval)],
                responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
                bbox=coords_bbox, size=coords_size, config=config
            )
        except Exception as e:
            logger.warn(e)
        
        data = None
        truecolor = None
        try:
            truecolor = request_truecolor.get_data()
        except Exception as e:
            logger.warning(e)

        image_true_color = None
        if len(truecolor) > 0:
           image_true_color = truecolor[0]
        if np.mean(image_true_color) != 0.0: #if image is not entirely blank
           
            image_filename = f"{interval}_{augmentation_type}_true_color.png"
            image_path = os.path.join(image_dir, "true_color", image_filename)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            save_image(image_true_color, image_path, factor=1.5/255, clip_range=(0, 1))
            logger.info(f"True Color Image saved to: {image_path}")

            
            data_name = str(interval).strip("()").replace("-","_").replace(",","_").replace("\'","").replace(" ","_") + "_" + augmentation_type + "_true_color"
            relative_path = f"sentinel/{tracer_id}/{job_id}/true_color/{data_name}.png"

        
            media_data = KernelPlancksterSourceData(
                name=data_name,
                protocol=protocol,
                relative_path=relative_path,
            )

            try:
                scraped_data_repository.register_scraped_photo(
                    job_id=job_id,
                    source_data=media_data,
                    local_file_name=image_path,
                )
            except Exception as e:
                logger.info("could not register file")

            # output_data_list.append(media_data)
           
        
    return output_data_list



def scrape(
    job_id: int,
    tracer_id: str,
    scraped_data_repository: ScrapedDataRepository,
    log_level: Logger,
    long_left: float,
    lat_down: float,
    long_right: float,
    lat_up: float,
    sentinel_config: SHConfig,
    start_date: str,
    end_date: str,
    image_dir: str,
    evalscript_bands_path: str,
    augmentation_type: str,
    resolution: int


) -> JobOutput:


    try:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=log_level)

        job_state = BaseJobState.CREATED
        current_data: KernelPlancksterSourceData | None = None
        last_successful_data: KernelPlancksterSourceData | None = None

        protocol = scraped_data_repository.protocol

        output_data_list: List[KernelPlancksterSourceData] = []
        if isinstance(sentinel_config, SHConfig):  # for typing
            config = sentinel_config
            # Set the job state to running
            logger.info(f"{job_id}: Starting Job")
            job_state = BaseJobState.RUNNING
            #job.touch()

            data = []



            
            start_time = time.time()  # Record start time for response time measurement
            try:
                # Create an instance of SentinelHubPipelineElement with the request data
                coords_wgs84 = (long_left,lat_down,long_right, lat_up)
                evalscript_bands_config = load_evalscript(evalscript_bands_path)
                logger.info(f"starting with augmentation_type: {augmentation_type}")
                output_data_list = get_images(logger, job_id, tracer_id, scraped_data_repository, output_data_list, protocol, coords_wgs84, evalscript_bands_config, config, start_date, end_date, resolution, image_dir, augmentation_type)
                output_data_list = augment_climate_images(image_dir,coords_wgs84) if augmentation_type == "climate" else augment_wildfire_images(image_dir,coords_wgs84)
                # Calculate response time
                response_time = time.time() - start_time
                response_data = {
                    "message": f"Pipeline processing completed",
                    "response_time": f"{response_time:.2f} seconds"
                }
        
            except Exception as e:
                logger.error(f"Error in processing pipeline: {e}")
                #raise HTTPException(status_code=500, detail="Internal server error occurred.")
                job_state = BaseJobState.FAILED
                logger.error(
                    f"{job_id}: Unable to scrape data. Error:\n{error}\nJob with tracer_id {tracer_id} failed.\nLast successful data: {last_successful_data}\nCurrent data: \"{current_data}\", job_state: \"{job_state}\""
                )
                #job.messages.append(f"Status: FAILED. Unable to scrape data. {error}")  # type: ignore
                #job.touch()

                # continue to scrape data if possible


                

            job_state = BaseJobState.FINISHED
            #job.touch()
            logger.info(f"{job_id}: Job finished")
            try:
                shutil.rmtree(image_dir)
            except Exception as e:
                logger.warning("Could not delete tmp directory, exiting")
            return JobOutput(
                job_state=job_state,
                tracer_id=tracer_id,
                source_data_list=output_data_list,
            )


    except Exception as error:
        logger.error(f"{job_id}: Unable to scrape data. Job with tracer_id {tracer_id} failed. Error:\n{error}")
        job_state = BaseJobState.FAILED
        try:
            logger.warning("deleting tmp directory")
            shutil.rmtree(image_dir)
        except Exception as e:
            logger.warning("Could not delete tmp directory, exiting")
        #job.messages.append(f"Status: FAILED. Unable to scrape data. {e}")