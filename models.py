from pydantic import BaseModel


class QueryModel(BaseModel):
    latitude: float
    longitude: float

class QueryModel(BaseModel):
    latitude: float
    longitude: float

class DataSourceModel(BaseModel):
    source: str
    q: QueryModel

class PipelineRequestModel(BaseModel):
    lfn: str
    # data_sources: list[DataSourceModel]


class SentinalHubRequest(PipelineRequestModel):
    lfn: str
    q: QueryModel