import json
import logging
import tempfile
from typing import List, Union

from app.config import SUPPORTED_DATASET_EVALSCRIPTS
from app.sdk.models import BaseJobState, JobOutput, KernelPlancksterSourceData, ProtocolEnum
from app.sdk.scraped_data_repository import ScrapedDataRepository
from app.time_travel.models import Error, Image, KeyFrame, Metadata, SentinelRowSchema
from utils import parse_relative_path


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def __filter_paths_by_timestamp(timestamp: str, relative_paths: List[KernelPlancksterSourceData]) -> List[str]:
    return [path.relative_path for path in relative_paths if timestamp in path.relative_path]

def generate_time_travel_metadata(
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
    case_study_name = "sentinel-5p"
    failed = False
    timestamps: List[str] = []
    relative_paths_for_agent: List[str] = []
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
            file_extension,
        ) = parse_relative_path(relative_path=relative_path)
        timestamps.append(timestamp)
        if file_extension in ["json", "csv", "txt"]:
            relative_paths_for_agent.append(relative_path)
    
    timestamps = list(set(timestamps))
    metadata: Metadata = Metadata(
        caseStudy="sentinel-5p",
        imageKinds=[] ,
        relativePathsForAgent=relative_paths_for_agent,
        keyframes=[],
    )
    for timestamp in timestamps:
        keyframe = KeyFrame(
            timestamp=timestamp,
            images=[],
            data=[],
            dataDescription=f"This data is a collection of Carbon Monoxide levels for the given timestamp for the given coordinates",
        )
        
        timestamp_relative_paths = __filter_paths_by_timestamp(timestamp, relevant_source_data)
        images_paths = [path for path in timestamp_relative_paths if path.endswith((".png", ".jpg", ".jpeg"))] 
        augmented_coordinates_path = [path for path in timestamp_relative_paths if path.endswith("augmented.json")]

        for image_path in images_paths:
            (
                _,
                _,
                _,
                timestamp,
                dataset,
                evalscript_name,
                _,
                file_extension,
            ) = parse_relative_path(relative_path=image_path)
            if dataset not in SUPPORTED_DATASET_EVALSCRIPTS:
                keyframe.images.append(Error(
                    errorMessage=f"Dataset {dataset} is not supported",
                    errorName="UnsupportedDataset",
                ))
                continue
            supported_eval_scripts = [x['name'] for x in SUPPORTED_DATASET_EVALSCRIPTS[dataset]["supported_evalscripts"]]
            if evalscript_name not in supported_eval_scripts:
                keyframe.images.append(Error(
                    errorMessage=f"Evalscript {evalscript_name} is not supported for {dataset}.",
                    errorName="UnsupportedEvalscript",
                ))
                continue
            if evalscript_name not in metadata.imageKinds:
                metadata.imageKinds.append(evalscript_name)
            evalscript = next((x for x in SUPPORTED_DATASET_EVALSCRIPTS[dataset]["supported_evalscripts"] if x['name'] == evalscript_name), None)
            if not evalscript:
                keyframe.images.append(Error(
                    errorMessage=f"Evalscript {evalscript_name} not found for {dataset}.",
                    errorName="MissingEvalscript",
                ))
                continue
            img_to_append = Image(
                relativePath=image_path,
                kind=evalscript_name,
                description=f"dataset: {dataset} | coords_wgs84: {long_left, lat_down, long_right, lat_up} | details: {evalscript['description']}",
            )
            keyframe.images.append(img_to_append)
     
        if len(augmented_coordinates_path) != 1:
            keyframe.data.append(Error(
                errorName="AugmentedCoordinatesError",
                errorMessage="Augmented data are missing or more than 1 dataset was found for this timestamp",
            ))
            metadata.keyframes.append(keyframe)
            continue
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as fp:
                scraped_data_repository.download_data(
                    source_data=KernelPlancksterSourceData(
                        name=augmented_coordinates_path[0].split("/")[-1],
                        protocol=protocol,
                        relative_path=augmented_coordinates_path[0],
                    ),
                    local_file=fp.name,
                )
                with open(fp.name, "r") as f:
                    augmented_coordinates: dict = json.load(f)
                for _, augmented_coordinate in augmented_coordinates.items():
                    keyframe.data.append(SentinelRowSchema(
                        timestamp=timestamp,
                        latitude=augmented_coordinate["latitude"],
                        longitude=augmented_coordinate["longitude"],
                        CarbonMonoxideLevel=augmented_coordinate["CO_level"],
                    ))
                metadata.keyframes.append(keyframe)
        except Exception as e:
            keyframe.data.append(Error(
                errorName="AugmentedCoordinatesError",
                errorMessage=f"Error while processing augmented coordinates: {e}",
            ))
            metadata.keyframes.append(keyframe)
    
    

    with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as out:
        with open(out.name, "w") as f:
            f.write(metadata.model_dump_json(indent=2))
        relative_path = f"{case_study_name}/{tracer_id}/{job_id}/sentinel/sentinel5p_time_travel_metadata.json"
        out_source_data = KernelPlancksterSourceData(
            name="time_travel_metadata.json",
            protocol=protocol,
            relative_path=relative_path,
        )
        try:
            scraped_data_repository.register_scraped_json(
                job_id=job_id,
                source_data=out_source_data,
                local_file_name=out.name,
            )
        except Exception as e:
            logger.error(f"Failed to upload time travel metadata: {e}")
            failed = True

    if failed:
        return JobOutput(
            job_state=BaseJobState.FAILED,
            tracer_id=tracer_id,
            source_data_list=[],
        )
    return JobOutput(
        job_state=BaseJobState.FINISHED,
        tracer_id=tracer_id,
        source_data_list=[out_source_data],
    )
    
