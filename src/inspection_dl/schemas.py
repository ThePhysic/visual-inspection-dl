"""Version 1 standalone inference contract (not the SWE platform's schema)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Box(ContractModel):
    x_min: float = Field(ge=0)
    y_min: float = Field(ge=0)
    x_max: float = Field(gt=0)
    y_max: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("box must have positive area")
        return self


class Detection(ContractModel):
    class_id: int = Field(ge=1)
    label: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    box: Box


class ImageShape(ContractModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Timing(ContractModel):
    decode_preprocess: float = Field(ge=0)
    model_inference: float = Field(ge=0)
    response_postprocess: float = Field(ge=0)
    request_total: float = Field(ge=0)


class Prediction(ContractModel):
    schema_version: Literal["1"] = "1"
    provider: Literal["visual-inspection-dl"] = "visual-inspection-dl"
    backend: Literal["trained", "fixture"]
    simulated: bool
    model_version: str
    taxonomy_version: str
    artifact_digest: str
    image: ImageShape
    coordinate_system: Literal["xyxy_pixels_original_image"] = "xyxy_pixels_original_image"
    confidence_threshold: float = Field(ge=0, le=1)
    nms_threshold: float = Field(ge=0, le=1)
    detections: list[Detection]
    timing_ms: Timing

    @model_validator(mode="after")
    def valid_geometry_and_backend(self):
        if self.simulated != (self.backend == "fixture"):
            raise ValueError("fixture provenance mismatch")
        for detection in self.detections:
            if detection.box.x_max > self.image.width or detection.box.y_max > self.image.height:
                raise ValueError("box exceeds original image bounds")
        return self
