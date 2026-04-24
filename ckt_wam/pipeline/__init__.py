"""CKT-WAM teacher--student pipeline wrappers."""

from ckt_wam.pipeline.ckt_pipeline import (
    CKTPipeline,
    CKTPipelineConfig,
    TeacherFeatureExtractor,
)
from ckt_wam.pipeline.ckt_pipeline_middle import (
    CKTPipelineMiddle,
    CKTPipelineMiddleConfig,
    TeacherMiddleLayerFeatureExtractor,
)

__all__ = [
    "CKTPipeline",
    "CKTPipelineConfig",
    "CKTPipelineMiddle",
    "CKTPipelineMiddleConfig",
    "TeacherFeatureExtractor",
    "TeacherMiddleLayerFeatureExtractor",
]
