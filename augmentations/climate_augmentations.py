from logging import Logger
import time
from app.sdk.models import KernelPlancksterSourceData, ProtocolEnum
from app.sdk.scraped_data_repository import ScrapedDataRepository,  KernelPlancksterSourceData
import numpy as np
import pandas as pd
import cv2
import os, re
from collections import Counter
import hashlib

def sanitize_filename(filename):   #helper function
    return re.sub(r'[^\w./]', '_', filename)

def dominant_color(pixels):        #helper function
    if len(pixels) == 0:
        return [0, 0, 0]
    color_counts = Counter(map(tuple, pixels))
    return max(color_counts, key=color_counts.get)

def get_image_hash(image):
    """
    Computes a hash for the given image.
    """
    hasher = hashlib.md5()
    hasher.update(image.tobytes())
    return hasher.hexdigest()

def augment_climate_images(case_study_name: str, job_id: str, tracer_id: str, image_dir: str, coords_wgs84: tuple[float, float, float, float], logger: Logger, protocol: ProtocolEnum, scraped_data_repository: ScrapedDataRepository, output_data_list: list[KernelPlancksterSourceData]):
    latitudes = [coords_wgs84[1], coords_wgs84[3]]
    longitudes = [coords_wgs84[0], coords_wgs84[2]]

    os.makedirs(os.path.join(image_dir, "masked"), exist_ok=True)
    for image_path in os.listdir(os.path.join(image_dir, "masked")):
        interval = "_".join(image_path.split("_")[:-1])
        image_hash = image_path.split("_")[-1].split(".")[0]
        full_path = os.path.join(image_dir, "masked", image_path)
        image = cv2.imread(full_path)
        height, width, _ = image.shape

        grid_size = 5
        grid_height = height // grid_size
        grid_width = width // grid_size

        data = []
        for grid_row in range(grid_size):
            for grid_col in range(grid_size):
                cell_pixels = image[grid_row * grid_height: (grid_row + 1) * grid_height,
                                    grid_col * grid_width: (grid_col + 1) * grid_width]
                cell_pixels = cell_pixels.reshape(-1, 3)
                dominant_pixel = dominant_color(cell_pixels)
                latitude = latitudes[0] + ((grid_row + 0.5) / grid_size) * (latitudes[1] - latitudes[0])
                longitude = longitudes[0] + ((grid_col + 0.5) / grid_size) * (longitudes[1] - longitudes[0])

                if np.array_equal(dominant_pixel, [127, 0, 0]):  # dark blue
                    data.append([latitude, longitude, "lowest-CO"])
                elif np.array_equal(dominant_pixel, [255, 0, 0]):  # blue
                    data.append([latitude, longitude, "low-CO"])
                elif np.array_equal(dominant_pixel, [255, 255, 0]):  # cyan
                    data.append([latitude, longitude, "moderately low-CO"])
                elif np.array_equal(dominant_pixel, [255, 255, 255]):  # yellow
                    data.append([latitude, longitude, "moderately high-CO"])
                elif np.array_equal(dominant_pixel, [0, 0, 255]):  # red
                    data.append([latitude, longitude, "high-CO"])
                elif np.array_equal(dominant_pixel, [0, 0, 127]):  # dark red
                    data.append([latitude, longitude, "highest-CO"])
                else:
                    data.append([latitude, longitude, "unknown"])

        if data:
            df = pd.DataFrame(data, columns=["latitude", "longitude", "CO_level"])
            jsonpath = os.path.join(image_dir, "augmented_coordinates", interval)
            os.makedirs(os.path.dirname(jsonpath), exist_ok=True)
            df.to_json(jsonpath, orient="index")
            logger.info(f"Augmented Climate Data saved to: {jsonpath}")

            # Sanitize the interval to create a valid filename
            sanitized_interval = sanitize_filename(interval)
            unix_timestamp = int(time.time())  # TODO: calculate a deterministic timestamp that can match those of the other scrapers given the same start_date, end_date, and interval

            data_name = f"{sanitized_interval}_climate_{image_hash}"
            relative_path = f"{case_study_name}/{tracer_id}/{job_id}/{unix_timestamp}/sentinel/augmented-coordinates/{data_name}.json"

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
                logger.warning(f"Could not register file: {e}")

            output_data_list.append(media_data)

    return output_data_list