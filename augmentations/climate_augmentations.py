import logging
import tempfile
from typing import List
from app.sdk.models import (
    BaseJobState,
    JobOutput,
    KernelPlancksterSourceData,
    ProtocolEnum,
)
from app.sdk.scraped_data_repository import (
    ScrapedDataRepository,
    KernelPlancksterSourceData,
)
import numpy as np
import pandas as pd
import cv2
from collections import Counter

from utils import generate_relative_path, parse_relative_path, sanitize_filename


def dominant_color(pixels):  # helper function
    if len(pixels) == 0:
        return [0, 0, 0]
    color_counts = Counter(map(tuple, pixels))
    return max(color_counts, key=color_counts.get)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def __filter_paths_by_timestamp(
    timestamp: str, relative_paths: List[KernelPlancksterSourceData]
) -> List[KernelPlancksterSourceData]:
    return [
        path for path in relative_paths if timestamp in path.relative_path
    ]


def augment_climate_images(
    case_study_name: str,
    job_id: int,
    tracer_id: str,
    long_left: float,
    lat_down: float,
    long_right: float,
    lat_up: float,
    scraped_data_repository: ScrapedDataRepository,
    relevant_source_data: list[KernelPlancksterSourceData],
    protocol: ProtocolEnum,
) -> JobOutput:
    failed = False
    latitudes = [lat_down, lat_up]
    longitudes = [long_left, long_right]
    timestamps: List[str] = []
    for source_data in relevant_source_data:
        relative_path = source_data.relative_path
        (
            _,
            _,
            _,
            timestamp,
            _,
            _,
            _,
            _,
        ) = parse_relative_path(relative_path=relative_path)
        timestamps.append(timestamp)

    timestamps = list(set(timestamps))
    for idx, timestamp in enumerate(timestamps):
        timestamp_relative_paths = __filter_paths_by_timestamp(
            timestamp, relevant_source_data
        )
        climate_mask_path: KernelPlancksterSourceData | None = next(
            (path for path in timestamp_relative_paths if "climate-mask" in path.relative_path), None
        )
        if climate_mask_path is None:
            logger.error(f"No climate mask data found for timestamp {timestamp}")
            continue
        (
            _,
            _,
            _,
            _,
            _,
            _,
            image_hash,
            _,
        ) = parse_relative_path(relative_path=climate_mask_path.relative_path)
        
        if image_hash == "empty":
            logger.error(f"Empty climate mask image found for timestamp {timestamp}")
            continue
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as fp:
            scraped_data_repository.download_data(
                source_data=climate_mask_path,
                local_file=fp.name
            )
            image = cv2.imread(fp.name)
            height, width, _ = image.shape

            grid_size = 5
            grid_height = height // grid_size
            grid_width = width // grid_size
            data = []
            for grid_row in range(grid_size):
                for grid_col in range(grid_size):
                    cell_pixels = image[
                        grid_row * grid_height : (grid_row + 1) * grid_height,
                        grid_col * grid_width : (grid_col + 1) * grid_width,
                    ]
                    cell_pixels = cell_pixels.reshape(-1, 3)
                    dominant_pixel = dominant_color(cell_pixels)
                    latitude = latitudes[0] + ((grid_row + 0.5) / grid_size) * (
                        latitudes[1] - latitudes[0]
                    )
                    longitude = longitudes[0] + ((grid_col + 0.5) / grid_size) * (
                        longitudes[1] - longitudes[0]
                    )

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

            if len(data) == 0:
                logger.error(f"No data found for climate mask image {fp.name}")
                continue
            df = pd.DataFrame(data, columns=["latitude", "longitude", "CO_level"])
            with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as out:
                df.to_json(out.name, orient="index")
                logger.info(
                    f"Augmented Climate Data locally saved to temporary file: {out.name}"
                )

                relative_path = generate_relative_path(
                    case_study_name=case_study_name,
                    tracer_id=tracer_id,
                    job_id=job_id,
                    timestamp=timestamp,
                    dataset='SENTINEL5P', #TODO: hardcoded value
                    evalscript_name='climate-mask', #TODO: hardcoded value
                    image_hash=image_hash + "-augmented",
                    file_extension="json",
                )

                media_data = KernelPlancksterSourceData(
                    name="augmented-coordinates.json",
                    protocol=protocol,
                    relative_path=relative_path,
                )

                try:
                    scraped_data_repository.register_scraped_json(
                        job_id=job_id,
                        source_data=media_data,
                        local_file_name=out.name,
                    )
                except Exception as e:
                    logger.error(f"Could not register file: {e}")
                    failed = True
                    continue

    if failed:
        return JobOutput(
            job_state=BaseJobState.FAILED,
            tracer_id=tracer_id,
            source_data_list=[],
        )
    return JobOutput(
        job_state=BaseJobState.FINISHED,
        tracer_id=tracer_id,
        source_data_list=[media_data],
    )
