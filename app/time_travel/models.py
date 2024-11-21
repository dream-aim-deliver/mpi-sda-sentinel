from pydantic import BaseModel
from typing import List, Literal, Union

class SentinelRowSchema(BaseModel):
    timestamp: str
    latitude: float
    longitude: float
    CarbonMonoxideLevel: str


class Error(BaseModel):
    errorName: str
    errorMessage: str

class Image(BaseModel):
    kind: str
    relativePath: str
    description: str


class KeyFrame(BaseModel):
    timestamp: str
    images: List[Union[Image, Error]]
    data: List[Union[SentinelRowSchema, Error]]
    dataDescription: str

class Metadata(BaseModel):
    caseStudy: Literal["sentinel-5p"]
    relativePathsForAgent: List[str]
    keyframes: List[KeyFrame]
    imageKinds: List[str]
