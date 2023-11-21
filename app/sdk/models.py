from enum import Enum
import random
import re
import string
from typing import List, TypeVar
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime


class QueryModel(BaseModel):
    """
    A model representing geographical coordinates.

    Attributes:
        latitude (float): The latitude of the location.
        longitude (float): The longitude of the location.
    """
    latitude: float
    longitude: float


class DataSourceModel(BaseModel):
    """
    A model representing a data source.

    Attributes:
        source (str): The name or identifier of the data source.
        q (QueryModel): The QueryModel instance containing latitude and longitude.
    """
    source: str
    q: QueryModel


class PipelineRequestModel(BaseModel):
    """
    A model representing a request for a pipeline process.

    Attributes:
        lfn (str): A local filename or identifier for the pipeline request.
        # data_sources (list[DataSourceModel]): A list of DataSourceModel instances (uncomment if needed).
    """
    lfn: str
    start_date: date = Field(default_factory=date.today)
    end_date: date = Field(default_factory=date.today)


class SentinelHubRequest(PipelineRequestModel):
    """
    A model extending PipelineRequestModel specifically for Sentinel Hub requests.

    Attributes:
        lfn (str): Inherits the local filename or identifier from PipelineRequestModel.
        q (QueryModel): A QueryModel instance representing geographical coordinates for the Sentinel Hub request.
    """
    q: QueryModel


class BaseJobState(Enum):
    CREATED = "created"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


class DataSource(Enum):
    TWITTER = "twitter"
    TELEGRAM = "telegram"
    SENTINEL = "sentinel"
    AUGMENTED_DATA = "augmented_data"


class Protocol(Enum):
    S3 = "s3"
    ES = "es"
    LOCAL = "local"


class LFN(BaseModel):
    protocol: Protocol
    tracer_id: str
    job_id: int
    source: DataSource
    relative_path: str

    @field_validator("relative_path")
    def relative_path_must_be_alphanumberic_underscores_backslashes(cls, v):
        marker = "sdamarker"
        if marker not in v:
            v = re.sub(r"[^a-zA-Z0-9_\./-]", "", v)
            ext = v.split(".")[-1]
            name = v.split(".")[0]
            seed = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
            v = f"{name}-{seed}-{marker}.{ext}"
        return v


class BaseJob(BaseModel):
    id: int
    tracer_id: str = Field(
        description="A unique identifier to trace jobs across the SDA runtime."
    )
    created_at: datetime = datetime.now()
    heartbeat: datetime = datetime.now()
    name: str
    args: dict = {}
    state: Enum = BaseJobState.CREATED
    messages: List[str] = []                  
    output_lfns: List[LFN] = []               
    input_lfns: List[LFN] = []                
    media_lfns: List[LFN] = []
    def touch(self) -> None:
        self.heartbeat = datetime.now()


TBaseJob = TypeVar("TBaseJob", bound=BaseJob)

