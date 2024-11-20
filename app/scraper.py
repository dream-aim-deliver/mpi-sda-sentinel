from oauthlib.oauth2.rfc6749.errors import InvalidClientError
from logging import Logger
from datetime import datetime,timedelta
import logging
from typing import List
from sentinelhub import SHConfig, BBox, CRS, DataCollection, SentinelHubRequest, bbox_to_dimensions, MimeType
from app.sdk.models import KernelPlancksterSourceData, BaseJobState, JobOutput, ProtocolEnum
from app.sdk.scraped_data_repository import ScrapedDataRepository,  KernelPlancksterSourceData
from app.setup import datetime_parser
from augmentations.climate_augmentations import augment_climate_images,get_image_hash
from augmentations.wildfire_augmentations import augment_wildfire_images, sanitize_filename
import time
import os
from utils import date_range, save_image
import numpy as np
import shutil
import requests

def load_evalscript(source: str) -> str:
    if source.startswith("http://") or source.startswith("https://"):
        # Load from URL
        response = requests.get(source)
        response.raise_for_status()  # Raise an exception if the request failed
        return response.text
    else:
        # Load from file
        with open(source, 'r') as file:
            return file.read()


def get_images(logger: Logger, case_study_name: str, job_id: int, tracer_id: str, scraped_data_repository: ScrapedDataRepository, 
            output_data_list: list[KernelPlancksterSourceData], protocol: ProtocolEnum, 
            coords_wgs84: tuple[float, float, float, float], evalscript_bands_config: str, evalscript_truecolor: str, config: SHConfig, start_date: str, end_date: str, 
            interval:int,
            resolution: int, image_dir: str, dataset_name: str):
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
        interval (int): Interval for data scraping
        resolution (int): Resolution for image retrieval.
        image_dir (str): Directory to save images.
        dataset_name (str): dataset applied.

    Returns:
        list: List of retrieved and processed images.
    """
    images = []
    coords_bbox = BBox(bbox=coords_wgs84, crs=CRS.WGS84)
    coords_size = bbox_to_dimensions(coords_bbox, resolution=resolution)
    current_date = datetime_parser(start_date) 
    last_date = datetime_parser(end_date)
    time_delta = timedelta(minutes=interval) #interval essentially
    dataset = None
    dataset_map = {
    "SENTINEL2_L1C": DataCollection.SENTINEL2_L1C,
    "SENTINEL5P": DataCollection.SENTINEL5P,
    # Add more datasets as needed
    }
    dataset = dataset_map.get(dataset_name)
    if not dataset:
        logger.error(f"Dataset {dataset_name} not supported. Use one of {dataset_map.keys()}")
        return output_data_list
    while current_date <= last_date :
        try:
            request_bands_config = SentinelHubRequest(
                evalscript=evalscript_bands_config,
                input_data=[SentinelHubRequest.input_data(data_collection=dataset, time_interval=(current_date.isoformat(), (current_date + time_delta).isoformat()))],
                responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
                bbox=coords_bbox, size=coords_size, config=config
            )
            data = request_bands_config.get_data()
        except InvalidClientError as e:
            logger.error(f"Sentinel Hub client error: {e}")
            raise e

        except Exception as e:
            logger.warning(e)
            continue

        if data:
            
            image = data[0]
            if np.mean(image) == 0.0:
                logger.warning("Image is blank, can't be registered, retrying for next interval")
                if not evalscript_truecolor : 
                    current_date += time_delta
            else:  # if image is not entirely blank
                image_hash = get_image_hash(image)
                file_extension = "png"
                try:
                    file_extension = image.format.lower()
                except:
                    pass
                
                image_filename = f"{dataset_name}_banded_config_{image_hash}.{file_extension}"
                image_path = os.path.join(image_dir, "banded_config", image_filename)
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                save_image(image, image_path, factor=1.5/255, clip_range=(0, 1))
                logger.info(f"Configured Bands Image saved to: {image_path}")
                data_name = sanitize_filename(f"{dataset_name}_banded_config_{image_hash}")
                unix_timestamp = int(current_date.timestamp()) 
                if not evalscript_truecolor : 
                    current_date += time_delta
                relative_path = f"{case_study_name}/{tracer_id}/{job_id}/{unix_timestamp}/sentinel/banded_config/{data_name}.{file_extension}"
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
                    logger.warning(f"Could not register file: {e}")

                output_data_list.append(media_data)

                image_filename = f"{dataset_name}_masked_{image_hash}.{file_extension}"
                image_path = os.path.join(image_dir, "masked", image_filename)
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                save_image(image, image_path, factor=255/255, clip_range=(0, 1))
                logger.info(f"Masked Image saved to: {image_path}")

                data_name = sanitize_filename(f"{dataset_name}_masked_{image_hash}")
                relative_path = f"{case_study_name}/{tracer_id}/{job_id}/{unix_timestamp}/sentinel/masked/{data_name}.{file_extension}"

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
                    logger.warning(f"Could not register file: {e}")

                output_data_list.append(media_data)
                
        if evalscript_truecolor:
            try:
                request_truecolor = SentinelHubRequest(
                evalscript=evalscript_truecolor,
                input_data=[SentinelHubRequest.input_data(data_collection=dataset, time_interval=(current_date.isoformat(), (current_date + time_delta).isoformat()))],
                responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
                bbox=coords_bbox, size=coords_size, config=config
            )
                truecolor = request_truecolor.get_data()
            except Exception as e:
                logger.warning(e)
                continue

        
            image_true_color = truecolor[0]
            if np.mean(image_true_color) == 0.0:
                logger.warning("Image is blank, can't be registered, retrying for next interval")
                current_date += time_delta
            else:  # if image is not entirely blank
                image_hash = get_image_hash(image_true_color)
                file_extension = "png"     
                try:
                    file_extension = image_true_color.format.lower()
                except:
                    pass
                image_filename = f"{interval}_{dataset_name}_true_color_{image_hash}.{file_extension}"
                image_path = os.path.join(image_dir, "true_color", image_filename)
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                save_image(image_true_color, image_path, factor=1.5/255, clip_range=(0, 1))
                logger.info(f"True Color Image saved to: {image_path}")
                current_date += time_delta
                data_name = sanitize_filename(f"{interval}_{dataset_name}_true_color_{image_hash}")
                relative_path = f"{case_study_name}/{tracer_id}/{job_id}/{unix_timestamp}/sentinel/true_color/{data_name}.{file_extension}"

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
                    logger.warning(f"Could not register file: {e}")
    return output_data_list


def scrape(
    case_study_name: str,
    job_id: int,
    tracer_id: str,
    scraped_data_repository: ScrapedDataRepository,
    log_level: str,
    long_left: float,
    lat_down: float,
    long_right: float,
    lat_up: float,
    sentinel_config: SHConfig,
    start_date: str,
    end_date: str,
    interval:str,
    image_dir: str,
    evalscript_bands_path: str,
    evalscript_truecolor_path:str,
    dataset_name: str,
    resolution: int
) -> JobOutput:


    try:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

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

            start_time = time.time()  # Record start time for response time measurement
            try:
                # Create an instance of SentinelHubPipelineElement with the request data
                coords_wgs84 = (long_left,lat_down,long_right, lat_up)
                evalscript_bands_config = load_evalscript(evalscript_bands_path)
                evalscript_truecolor = load_evalscript(evalscript_truecolor_path) if evalscript_truecolor_path else False
                logger.info(f"starting with dataset: {dataset_name}")
                output_data_list = get_images(logger, case_study_name, job_id, tracer_id, scraped_data_repository, output_data_list, protocol, coords_wgs84, evalscript_bands_config, evalscript_truecolor ,config, start_date, end_date, interval, resolution, image_dir, dataset_name)
                output_data_list = augment_wildfire_images(case_study_name, job_id, tracer_id, image_dir, coords_wgs84, logger, protocol, scraped_data_repository,output_data_list) if dataset_name == "SENTINEL2_L1C" else augment_climate_images(case_study_name, job_id, tracer_id, image_dir, coords_wgs84, logger, protocol, scraped_data_repository,output_data_list) 

                # Calculate response time
                response_time = time.time() - start_time
                response_data = {
                    "message": f"Pipeline processing completed",
                    "response_time": f"{response_time:.2f} seconds"
                }

                job_state = BaseJobState.FINISHED
                logger.info(f"{job_id}: Job finished")
        
            except Exception as e:
                logger.error(f"Error in processing pipeline: {e}")
                job_state = BaseJobState.FAILED
                logger.error(
                    f"{job_id}: Unable to scrape data. Error: {e}\nJob with tracer_id {tracer_id} failed."
                )
                logger.error(
                    f"Last successful data: {last_successful_data} -- Current data: \"{current_data}\" -- job_state: \"{job_state}\""
                )

            finally:
                try:
                    shutil.rmtree(image_dir)
                except Exception as e:
                    logger.warning(f"Could not delete tmp directory, exiting: {e}")
                
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
            logger.warning(f"Could not delete tmp directory, exiting: {e}")

        return JobOutput(
            job_state=job_state,
            tracer_id=tracer_id,
            source_data_list=[],
        )
        
        