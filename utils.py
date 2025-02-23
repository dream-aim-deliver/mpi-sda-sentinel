from oauthlib.oauth2.rfc6749.errors import InvalidClientError
from typing import Any, NamedTuple, Optional, Tuple
import requests
# Matplotlib is not thread safe, so we need to set the backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from logging import Logger
import hashlib
import re
import numpy as np
from PIL import Image
from typing import Literal

from sentinelhub import (
    SHConfig,
    BBox,
    CRS,
    DataCollection,
    SentinelHubRequest,
    bbox_to_dimensions,
    MimeType,
)
def plot_image(
    image: np.ndarray, factor: float = 1.0, clip_range: Optional[Tuple[float, float]] = None, **kwargs: Any
) -> None:
    """Utility function for plotting RGB images."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().set_axis_off()
    ax.set_frame_on(False)

def download_image(
    logger: Logger,
    coords_wgs84: tuple[float, float, float, float],
    dataset: DataCollection,
    start_time: datetime,
    end_time: datetime,
    evalscript: str,
    config: SHConfig,
    resolution: int,
):
    coords_bbox = BBox(coords_wgs84, crs=CRS.WGS84)
    coords_size = bbox_to_dimensions(coords_bbox, resolution=resolution)
    try:
        sentinel_request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=dataset,
                    time_interval=(start_time, end_time),
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
            bbox=coords_bbox,
            size=coords_size,
            config=config,
        )
        data = sentinel_request.get_data()
        return data
    except InvalidClientError as e:
        logger.error(f"Sentinel Hub client error: {e}")

    except Exception as e:
        logger.warning(e)

def save_image(
    image: np.ndarray, filename: str, factor: float = 1.0, clip_range: Optional[Tuple[float, float]] = None, **kwargs: Any
) -> None:
    """Utility function for saving RGB images to a file."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().set_axis_off()
    plt.margins(0,0)
    plt.savefig(filename, bbox_inches="tight",pad_inches = 0)
    plt.close()


def generate_empty_image():
    """Utility function for generating an empty RGB image."""
    return np.zeros((256, 256, 3), dtype=np.uint8)


def is_image_empty(image: np.ndarray) -> bool:
    """ Utility function for checking if an image is empty."""
    img = Image.fromarray(image)
    if img.mode == "RGBA":
        alpha_channel = np.array(img.getchannel("A"))
        transparent_ratio = np.sum(alpha_channel == 0) / alpha_channel.size
        if transparent_ratio > 0.9:
            # if np.all(alpha_channel == 0):
            return True

    img_gray = img.convert("L")
    img_array = np.array(img_gray)

    # Check if the image is entirely black, natural
    if np.all(img_array == 0):
        print(f"Image is empty (black)")
        return True

    # Check if the image is entirely white, natural
    elif np.all(img_array == 255):
        print(f"Image is empty (white)")
        return True
    return False

def get_image_hash(image):
    """
    Computes a hash for the given image.
    """
    hasher = hashlib.md5()
    hasher.update(image.tobytes())
    return hasher.hexdigest()


def date_range(start_date, end_date):
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    date_list = [((start_date + timedelta(days=x)).strftime("%Y-%m-%d"),
                  (start_date + timedelta(days=x+1)).strftime("%Y-%m-%d"))
                 for x in range((end_date-start_date).days + 1)]
    return date_list


def load_evalscript(source: str) -> str:
    if source.startswith("http://") or source.startswith("https://"):
        # Load from URL
        response = requests.get(source)
        response.raise_for_status()  # Raise an exception if the request failed
        return response.text
    else:
        # Load from file
        with open(source, "r") as file:
            return file.read()


def sanitize_filename(filename):   #helper function
    return re.sub(r'[^\w./]', '_', filename)


class KernelPlancksterRelativePath(NamedTuple):
    case_study_name: str
    tracer_id: str
    job_id: str
    timestamp: str
    dataset: str
    evalscript_name: str
    image_hash: str
    file_extension: str

def generate_relative_path(case_study_name, tracer_id, job_id, timestamp, dataset, evalscript_name, image_hash, file_extension):
    return f"{case_study_name}/{tracer_id}/{job_id}/{timestamp}/sentinel/{dataset}_{evalscript_name}_{image_hash}.{file_extension}"

def parse_relative_path(relative_path) -> KernelPlancksterRelativePath:
    parts = relative_path.split("/")
    case_study_name = parts[0]
    tracer_id = parts[1]
    job_id = parts[2]
    timestamp = parts[3]
    dataset, evalscript_name, image_hash_extension = parts[5].split("_")
    image_hash, file_extension = image_hash_extension.split(".")
    return KernelPlancksterRelativePath(case_study_name, tracer_id, job_id, timestamp, dataset, evalscript_name, image_hash, file_extension)