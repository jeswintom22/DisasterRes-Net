"""One inference stack per process; serialise GPU/model calls on Windows."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from damage_assessment.localization import DamageLocalizationAnalyzer
from models.hybrid_pipeline import HybridDisasterPipeline

_pipeline: HybridDisasterPipeline | None = None
_analyzer: DamageLocalizationAnalyzer | None = None
_lock = Lock()
INFERENCE_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="disaster-inference")

def get_pipeline() -> HybridDisasterPipeline:
    global _pipeline
    with _lock:
        if _pipeline is None:
            _pipeline = HybridDisasterPipeline()
        return _pipeline

def get_damage_analyzer() -> DamageLocalizationAnalyzer:
    global _analyzer
    with _lock:
        if _analyzer is None:
            _analyzer = DamageLocalizationAnalyzer()
        return _analyzer

def warm() -> None:
    """Construct lightweight pipeline objects and force CUDA/cuDNN init."""
    import numpy as np
    pipeline = get_pipeline()
    get_damage_analyzer()
    pipeline.analyze(np.zeros((299, 299, 3), dtype=np.uint8))
