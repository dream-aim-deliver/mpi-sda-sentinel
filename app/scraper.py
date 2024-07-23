from logging import Logger
import logging
from typing import List
from sentinelhub import SHConfig, BBox, CRS, DataCollection, SentinelHubRequest, bbox_to_dimensions, MimeType
from app.sdk.models import KernelPlancksterSourceData, BaseJobState, JobOutput, ProtocolEnum
from app.sdk.scraped_data_repository import ScrapedDataRepository,  KernelPlancksterSourceData
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

def get_satellite_bands_config(augmentation_type: str) -> str:
        """
        Returns the evalscript configuration for Sentinel Hub.

        Returns:
            str: Evalscript for true color imagery.
        """
        if augmentation_type == "wildfire":
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
        
        elif augmentation_type == "climate":
            return """

            //VERSION=3
            var minVal = 0.0;
            var maxVal = 0.1;
            var diff = maxVal - minVal;
            const map = [
                [minVal, 0x00007f], 
                [minVal + 0.125 * diff, 0x0000ff],
                [minVal + 0.375 * diff, 0x00ffff],
                [minVal + 0.625 * diff, 0xffff00],
                [minVal + 0.875 * diff, 0xff0000],
                [maxVal, 0x7f0000]
            ]; 

            const visualizer = new ColorRampVisualizer(map)
            function setup() {
            return {
                input: ["CO","dataMask"],
                output: { bands: 4 }
            };
            }

            function evaluatePixel(samples) {
            const [r, g, b] = visualizer.process(samples.CO);
            return [r, g, b, samples.dataMask];
            }
            """

def get_true_color_config():
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


def get_images(logger: Logger, job_id: int, tracer_id:str, scraped_data_repository: ScrapedDataRepository, output_data_list: list[KernelPlancksterSourceData], protocol: ProtocolEnum, coords_wgs84: tuple[float, float, float, float], evalscript_bands_config: str, config: SHConfig, start_date: str, end_date: str, resolution: int, image_dir: str, augmentation_type: str):
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
    coords_bbox = BBox(bbox=coords_wgs84, crs=CRS.WGS84)
    coords_size = bbox_to_dimensions(coords_bbox, resolution=resolution)
    date_intervals = date_range(start_date, end_date)
    evalscript_truecolor = get_true_color_config()
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
                input_data=[SentinelHubRequest.input_data(data_collection= DataCollection.SENTINEL2_L1C, time_interval=interval)],
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


        

def augment_images(logger: Logger, job_id: int, tracer_id:str, scraped_data_repository: ScrapedDataRepository, output_data_list: list[KernelPlancksterSourceData], protocol: ProtocolEnum, coords_wgs84: tuple[float, float, float, float], image_dir: str, augmentation_type: str):
    # Read the satellite image
    latitudes = [coords_wgs84[1], coords_wgs84[3]]  
    longitudes = [coords_wgs84[0], coords_wgs84[2]]

    for image_path in os.listdir(os.path.join(image_dir,"masked")):
        full_path = os.path.join(image_dir, "masked", image_path)
        image = cv2.imread(full_path)
        # Extract image dimensions
        height, width, _ = image.shape
        data = []
        # Loop through the image and check for pure red pixels
        for i in range(height):
            for j in range(width):
                # Extract pixel values
                pixel = image[i, j]
                if augmentation_type == "wildfire":
                    if (pixel == [0, 0, 255]).all(): #bgr
                        # Convert pixel coordinates to latitude and longitude
                        latitude = latitudes[0] + (i / height) * (latitudes[1] - latitudes[0])
                        longitude = longitudes[0] + (j / width) * (longitudes[1] - longitudes[0])
                        # Add a row to the DataFrame
                        data.append([latitude, longitude, "forestfire"])
                elif augmentation_type == "climate":
                    if (pixel == [0, 0, 255]).all(): #bgr
                        # Convert pixel coordinates to latitude and longitude
                        latitude = latitudes[0] + (i / height) * (latitudes[1] - latitudes[0])
                        longitude = longitudes[0] + (j / width) * (longitudes[1] - longitudes[0])
                        # Add a row to the DataFrame
                        data.append([latitude, longitude, "climate"])
        
        #TODO: implement tempfile
        df = pd.DataFrame(data, columns=['latitude', 'longitude', 'status'])
        jsonpath = os.path.join(image_dir, "augmented_coordinates", image_path)
        os.makedirs(os.path.dirname(jsonpath), exist_ok=True)
        df.to_json(jsonpath, orient="index") 
        logger.info(f"Augmented JSON saved to: {jsonpath}")
        data_name = str(os.path.splitext(image_path)[0]).strip("(").replace(")","").replace("-","_").replace(",","_").replace("\'","").replace(" ","_")

        relative_path = f"sentinel/{tracer_id}/{job_id}/augmented/{data_name}.json"

    
        media_data = KernelPlancksterSourceData(
            name=data_name,
            protocol=protocol,
            relative_path=relative_path,
        )

        
        try:
            scraped_data_repository.register_scraped_json(
                job_id=job_id,
                source_data=media_data,
                local_file_name=jsonpath,
            )
        except Exception as e:
            logger.info("could not register file")

        output_data_list.append(media_data)
        #job.touch()
    
     

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
                evalscript_bands_config = get_satellite_bands_config(augmentation_type=augmentation_type)
                output_data_list = get_images(logger, job_id, tracer_id, scraped_data_repository, output_data_list, protocol, coords_wgs84, evalscript_bands_config, config, start_date, end_date, resolution, image_dir, augmentation_type)
                output_data_list = augment_images(logger, job_id, tracer_id, scraped_data_repository, output_data_list, protocol, coords_wgs84, image_dir, augmentation_type)

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