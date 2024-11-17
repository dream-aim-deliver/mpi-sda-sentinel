from enum import Enum
import os
import re
from typing import List, TypeVar
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class BaseJobState(Enum):
    CREATED = "created"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"

class ProtocolEnum(Enum):
    """
    The storage protocol to use for a file.

    Attributes:
    - S3: S3
    - LOCAL: Local  @deprecated
    """
    S3 = "s3"
    LOCAL = "local"


class KernelPlancksterSourceData(BaseModel):
    """
    Synchronize this with Kernel Planckster's SourceData model, so that this client generates valid requests.

    @attr name: the name of the source data to register as metadata
    @attr protocol: the protocol to use to store the source data
    @attr relative_path: the relative path to store the source data in the storage system
    """
    name: str
    protocol: ProtocolEnum
    relative_path: str

    def to_json(cls) -> str:
        """
        Dumps the model to a json formatted string. Wrapper around pydantic's model_dump_json method: in case they decide to deprecate it, we only refactor here.
        """
        return cls.model_dump_json()

    def __str__(self) -> str:
        return self.to_json()

    @classmethod
    def from_json(cls, json_str: str) -> "KernelPlancksterSourceData":
        """
        Loads the model from a json formatted string. Wrapper around pydantic's model_validate_json method: in case they decide to deprecate it, we only refactor here.
        """
        return cls.model_validate_json(json_data=json_str)

    @classmethod
    def name_validation(cls, v: str) -> str:
        if v == "":
            raise ValueError("The name must not be empty")
        return v

    @classmethod
    def relative_path_validation(cls, v: str) -> str:
        value_error_flag = False
        value_error_msg = ""

        if v == "":
            value_error_msg += f"The relative path must not be empty. "
            raise ValueError(value_error_msg)

        v2 = re.sub(r"[^a-zA-Z0-9_\./-]", "", v)
        if v != v2:
            value_error_flag = True
            value_error_msg += f"The relative path must contain only alphanumeric characters, underscores, slashes, and dots. Other characters are not allowed. "

        ext = os.path.splitext(v)[1].replace(".", "")
        if ext == "":
            value_error_flag = True
            value_error_msg += f"The relative path provided did not have an extension. Extensions are required to infer the type of the source data. "

        first_char = v[0]
        if first_char == "/":
            value_error_flag = True
            value_error_msg += f"The relative path provided must not start with a slash. "

        if value_error_flag:
            value_error_msg += f"\nThe relative path provided was: '{v}'"
            raise ValueError(value_error_msg)

        return v

    @classmethod
    def protocol_validation(cls, v: str) -> ProtocolEnum:
        all_protocols = [e for e in ProtocolEnum]
        all_protocols_str = [p.value for p in all_protocols]
        implemented_protocols = [ProtocolEnum.S3]
        implemented_protocols_str = [p.value for p in implemented_protocols]

        try:
            enum = ProtocolEnum(v)
        except ValueError:
            raise ValueError(
                f"'{v}' is not a valid protocol. Valid protocols are:\n{all_protocols_str}\nImplemented protocols are:\n{implemented_protocols_str}"
            )

        if enum not in implemented_protocols:
            raise ValueError(
                f"The protocol '{v}' is not implemented. Please use one of the following: {implemented_protocols_str}"
            )

        return ProtocolEnum(v)

    @field_validator("name")
    def name_must_not_be_empty(cls, v: str) -> str:
        return cls.name_validation(v)

    @field_validator("relative_path")
    def relative_path_must_be_correctly_formatted(cls, v: str) -> str:
        return cls.relative_path_validation(v)

    @field_validator("protocol")
    def protocol_must_be_supported(cls, v: ProtocolEnum) -> ProtocolEnum:
        return cls.protocol_validation(v.value)


class BaseJob(BaseModel):
    """
    NOTE: deprecated.
    """
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
    output_source_data_list: List[KernelPlancksterSourceData] = []
    input_source_data_list: List[KernelPlancksterSourceData] = []

    def touch(self) -> None:
        self.heartbeat = datetime.now()


TBaseJob = TypeVar("TBaseJob", bound=BaseJob)


class JobOutput(BaseModel):
    """
    This class is used to represent the output of a scraper job.

    Attributes:
    - job_state: BaseJobState
    - trace_id: str
    - source_data_list: List[KernelPlancksterSourceData] | None
    """

    job_state: BaseJobState
    tracer_id: str
    source_data_list: List[KernelPlancksterSourceData] | None

