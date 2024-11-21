import logging
import tempfile
from app.sdk.models import BaseJobState, JobOutput, KernelPlancksterSourceData, ProtocolEnum
from app.sdk.scraped_data_repository import ScrapedDataRepository,  KernelPlancksterSourceData
import pandas as pd
import cv2

from utils import generate_relative_path, parse_relative_path


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def augment_wildfire(
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
    for source_data in relevant_source_data:
        relative_path = source_data.relative_path
        (
            case_study_name,
            tracer_id,
            job_id,
            timestamp,
            dataset,
            evalscript_name,
            image_hash,
            file_extension,
        ) = parse_relative_path(relative_path=relative_path)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as fp:
            scraped_data_repository.download_data(
                source_data=source_data, local_file=fp.name
            )
            image = cv2.imread(fp.name)
            height, width, _ = image.shape

        data = []
        for i in range(height):
            for j in range(width):
                pixel = image[i, j]
                if (pixel == [0, 0, 255]).all():  # bgr
                    latitude = latitudes[0] + (i / height) * (latitudes[1] - latitudes[0])
                    longitude = longitudes[0] + (j / width) * (longitudes[1] - longitudes[0])
                    data.append([latitude, longitude, "forestfire"])

        if len(data) == 0:
            logger.error(f"No data found for image {fp.name}")
            continue

        df = pd.DataFrame(data, columns=['latitude', 'longitude', 'status'])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as out:
            df.to_json(out.name, orient="index")
            logger.info(
                f"Augmented Data locally saved to temporary file: {out.name}"
            )

            relative_path = generate_relative_path(
                    case_study_name=case_study_name,
                    tracer_id=tracer_id,
                    job_id=job_id,
                    timestamp=timestamp,
                    dataset=dataset,
                    evalscript_name=evalscript_name,
                    image_hash=image_hash + "-augmented",
                    file_extension="json"
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