from logging import Logger
from app.sdk.models import KernelPlancksterSourceData, ProtocolEnum
from app.sdk.scraped_data_repository import ScrapedDataRepository,  KernelPlancksterSourceData
import pandas as pd
import cv2
import os, re
import hashlib

def sanitize_filename(filename):
    # Replace disallowed characters with underscores
    return re.sub(r'[^\w./]', '_', filename)

def get_image_hash(image):
    """
    Computes a hash for the given image.
    """
    hasher = hashlib.md5()
    hasher.update(image.tobytes())
    return hasher.hexdigest()

def augment_wildfire_images(job_id: str, tracer_id: str, image_dir: str, coords_wgs84: tuple[float, float, float, float], logger: Logger, protocol: ProtocolEnum, scraped_data_repository: ScrapedDataRepository, output_data_list: list[KernelPlancksterSourceData]):
    latitudes = [coords_wgs84[1], coords_wgs84[3]]
    longitudes = [coords_wgs84[0], coords_wgs84[2]]

    os.makedirs(os.path.join(image_dir, "masked"), exist_ok=True)
    for image_path in os.listdir(os.path.join(image_dir, "masked")):
        interval = "_".join(image_path.split("_")[:-1])
        image_hash = image_path.split("_")[-1].split(".")[0]
        full_path = os.path.join(image_dir, "masked", image_path)
        image = cv2.imread(full_path)
        height, width, _ = image.shape

        data = []
        for i in range(height):
            for j in range(width):
                pixel = image[i, j]
                if (pixel == [0, 0, 255]).all():  # bgr
                    latitude = latitudes[0] + (i / height) * (latitudes[1] - latitudes[0])
                    longitude = longitudes[0] + (j / width) * (longitudes[1] - longitudes[0])
                    data.append([latitude, longitude, "forestfire"])

        if data:
            df = pd.DataFrame(data, columns=['latitude', 'longitude', 'status'])
            jsonpath = os.path.join(image_dir, "augmented_coordinates", interval)
            os.makedirs(os.path.dirname(jsonpath), exist_ok=True)
            df.to_json(jsonpath, orient="index")
            logger.info(f"Augmented JSON saved to: {jsonpath}")

            # Sanitize the interval to create a valid filename
            sanitized_interval = sanitize_filename(interval)

            data_name = sanitize_filename(f"{sanitized_interval}_wildfire_{image_hash}")
            relative_path = f"sentinel/{tracer_id}/{job_id}/augmented-coordinates/{data_name}.json"

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

    return output_data_list

