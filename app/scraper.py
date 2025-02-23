from datetime import timedelta
import logging
from typing import Any, List
from sentinelhub import (
    SHConfig,
)
from app.sdk.models import (
    KernelPlancksterSourceData,
    BaseJobState,
    JobOutput,
)
from app.sdk.scraped_data_repository import (
    ScrapedDataRepository,
    KernelPlancksterSourceData,
)
from app.setup import datetime_parser

from utils import download_image, generate_empty_image, generate_relative_path, get_image_hash, is_image_empty, load_evalscript, save_image
import tempfile


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
    interval: int,
    dataset_evalscripts: dict[str, dict[str, Any]],
    resolution: int,
    insert_empty_images: bool,
) -> JobOutput:

    try:
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        job_state = BaseJobState.CREATED
        current_data: KernelPlancksterSourceData | None = None
        last_successful_data: KernelPlancksterSourceData | None = None

        protocol = scraped_data_repository.protocol

        output_data_list: List[KernelPlancksterSourceData] = []
        if not isinstance(sentinel_config, SHConfig):
            return JobOutput(
                job_state=BaseJobState.FAILED,
                tracer_id=tracer_id,
                source_data_list=[],
            )

        # Set the job state to running
        logger.info(f"{job_id}: Starting Job")
        job_state = BaseJobState.RUNNING

        try:
            # Create an instance of SentinelHubPipelineElement with the request data
            coords_wgs84 = (long_left, lat_down, long_right, lat_up)
            parsed_start_date = datetime_parser(start_date)
            parsed_end_date = datetime_parser(end_date)
            parsed_interval = timedelta(minutes=interval)
            current_datetime = parsed_start_date
            current_iteration = 1
            total_iterations = int(
                (parsed_end_date - parsed_start_date) / parsed_interval
            ) + 1
            while current_datetime <= parsed_end_date:
                logger.info(
                    f"Processing iteration {current_iteration} of {total_iterations}"
                )
                datasets = dataset_evalscripts.keys()
                for dataset in datasets:
                    sentinel_dataset = dataset_evalscripts[dataset]["sentinel_sdk_obj"]
                    evalscripts = dataset_evalscripts[dataset]["evalscripts"]
                    for evalscript_config in evalscripts:
                        logger.info(f"{current_iteration}/{total_iterations}: Processing evalscript {evalscript_config['name']}")
                        evalscript_name = evalscript_config["name"]
                        evalscript_path = evalscript_config["path"]
                        evalscript = load_evalscript(evalscript_path)
                        logger.info(f"{current_iteration}/{total_iterations} Downloading image for evalscript from {evalscript_config['name']}")
                        image = download_image(
                            logger,
                            coords_wgs84,
                            sentinel_dataset,
                            current_datetime,
                            current_datetime + parsed_interval,
                            evalscript,
                            sentinel_config,
                            resolution,
                        )
                        if not image or len(image) == 0:
                            logger.warning(f"{current_iteration}/{total_iterations} No image found!")
                            if insert_empty_images:
                                image = generate_empty_image()
                                image = [image]
                            else:
                                continue
                        image = image[0]
                        if is_image_empty(image):
                            image_hash = "empty"
                        else:
                            image_hash = get_image_hash(image)
                        file_extension = "png"
                        clip_range = evalscript_config.get("clip_range", (0, 1))
                        scaling_factor = evalscript_config.get("scaling_factor", 1.5) / 255
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as fp:
                            logger.info(f"{current_iteration}/{total_iterations} Saving image for evalscript from {evalscript_config['name']}")
                            save_image(image=image, filename=fp.name, factor=scaling_factor, clip_range=(clip_range['min'], clip_range['max']))
                            file_name = f"{dataset}_{evalscript_name}_{image_hash}"
                            relative_path = generate_relative_path(case_study_name=case_study_name, tracer_id=tracer_id, job_id=job_id, timestamp=int(current_datetime.timestamp()), dataset=dataset, evalscript_name=evalscript_name, image_hash=image_hash, file_extension=file_extension)
                            source_data = KernelPlancksterSourceData(
                                name=file_name,
                                protocol=protocol,
                                relative_path=relative_path,
                            )
                            try:
                                logger.info(f"{current_iteration}/{total_iterations} Registering image for evalscript from {evalscript_config['name']}")
                                scraped_data_repository.register_scraped_photo(
                                    source_data=source_data,
                                    job_id=job_id,
                                    local_file_name=fp.name,
                                )
                                output_data_list.append(source_data)
                            except Exception as e:
                                logger.warning(f"{current_iteration}/{total_iterations} Could not register file {source_data}: {e}")
                                continue
                current_datetime += parsed_interval
                current_iteration += 1

            job_state = BaseJobState.FINISHED
            logger.info(f"{job_id}: Job finished")

        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}")
            job_state = BaseJobState.FAILED
            logger.error(
                f"{job_id}: Unable to scrape data. Error: {e}\nJob with tracer_id {tracer_id} failed."
            )
            logger.error(
                f'Last successful data: {last_successful_data} -- Current data: "{current_data}" -- job_state: "{job_state}"'
            )

        finally:
            return JobOutput(
                job_state=job_state,
                tracer_id=tracer_id,
                source_data_list=output_data_list,
            )

    except Exception as error:
        logger.error(
            f"{job_id}: Unable to scrape data. Job with tracer_id {tracer_id} failed. Error:\n{error}"
        )
        job_state = BaseJobState.FAILED
        return JobOutput(
            job_state=job_state,
            tracer_id=tracer_id,
            source_data_list=[],
        )
