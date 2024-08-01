from logging import Logger
import logging
from typing import List
from sentinelhub import SHConfig, BBox, CRS, DataCollection, SentinelHubRequest, bbox_to_dimensions, MimeType
from app.sdk.models import KernelPlancksterSourceData, BaseJobState, JobOutput, ProtocolEnum
from app.sdk.scraped_data_repository import ScrapedDataRepository,  KernelPlancksterSourceData
import json
import pandas as pd
import cv2
import os , re

def sanitize_filename(filename):
    # Replace disallowed characters with underscores
    return re.sub(r'[^\w./]', '_', filename)

def augment_climate_images(job_id:str, tracer_id:str ,image_dir:str, coords_wgs84:tuple[float,float,float,float], logger:Logger, protocol:ProtocolEnum, scraped_data_repository : ScrapedDataRepository , output_data_list: list[KernelPlancksterSourceData]):
    latitudes = [coords_wgs84[1], coords_wgs84[3]]  
    longitudes = [coords_wgs84[0], coords_wgs84[2]]

    for image_path in os.listdir(os.path.join(image_dir, "masked")):
        interval = os.path.splitext(image_path)[0]
        full_path = os.path.join(image_dir, "masked", image_path)
        image = cv2.imread(full_path)
        # Extract image dimensions
        height, width, _ = image.shape
        data = []
        for i in range(height):
            for j in range(width):
                pixel = image[i, j]
                if (pixel == [127,0,0]).all():  # dark blue
                    latitude = latitudes[0] + (i / height) * (latitudes[1] - latitudes[0])
                    longitude = longitudes[0] + (j / width) * (longitudes[1] - longitudes[0])
                    data.append([latitude, longitude, "lowest-CO"])
                elif (pixel == [255,0,0]).all():  # blue
                    latitude = latitudes[0] + (i / height) * (latitudes[1] - latitudes[0])
                    longitude = longitudes[0] + (j / width) * (longitudes[1] - longitudes[0])
                    data.append([latitude, longitude, "low-CO"])
                elif (pixel == [255,255,0]).all(): # cyan
                    latitude = latitudes[0] + (i / height) * (latitudes[1] - latitudes[0])
                    longitude = longitudes[0] + (j / width) * (longitudes[1] - longitudes[0])
                    data.append([latitude, longitude, "moderately low-CO"])
                elif (pixel == [255,255,255]).all(): #yellow
                    latitude = latitudes[0] + (i / height) * (latitudes[1] - latitudes[0])
                    longitude = longitudes[0] + (j / width) * (longitudes[1] - longitudes[0])
                    data.append([latitude, longitude, "moderately high-CO"])
                elif (pixel == [0,0,255]).all(): #red
                    latitude = latitudes[0] + (i / height) * (latitudes[1] - latitudes[0])
                    longitude = longitudes[0] + (j / width) * (longitudes[1] - longitudes[0])
                    data.append([latitude, longitude, "high-CO"])
                elif (pixel == [0,0,127]).all(): #dark red
                    latitude = latitudes[0] + (i / height) * (latitudes[1] - latitudes[0])
                    longitude = longitudes[0] + (j / width) * (longitudes[1] - longitudes[0])
                    data.append([latitude, longitude, "highest-CO"])

        # Save data to JSON
        try:
            if data:
                df = pd.DataFrame(data, columns=['latitude', 'longitude', 'status'])
                jsonpath = os.path.join(image_dir, "augmented_coordinates", interval)
                os.makedirs(os.path.dirname(jsonpath), exist_ok=True)
                df.to_json(jsonpath, orient="index") 
                logger.info(f"Augmented JSON saved to: {jsonpath}")

                # Sanitize the interval to create a valid filename
                sanitized_interval = sanitize_filename(interval)
                
                data_name = str(sanitized_interval).strip("()").replace("-", "_").replace(",", "_").replace("'", "").replace(" ", "_")
                relative_path = f"sentinel/{tracer_id}/{job_id}/augmented/{data_name}.json"
        

    
                media_data = KernelPlancksterSourceData(
                              name=data_name,
                              protocol=protocol,
                              relative_path=relative_path,
                            )
                
        except Exception as e:
            logger.error(e)

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
