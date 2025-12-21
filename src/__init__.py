"""
Historical Manuscript Processing Pipeline

Пакет для автоматического распознавания символов 
в кириллических исторических рукописях.
"""

__version__ = "1.0.0"
__author__ = "Andrey Nikishin"

from .image_processor import ImageProcessor
from .corpus_detector import CorpusDetector
from .line_segmenter import LineSegmenter
from .segmentation_visualizer import SegmentationVisualizer
from .inference_pipeline import InferencePipeline

__all__ = [
    "ImageProcessor",
    "CorpusDetector",
    "LineSegmenter",
    "SegmentationVisualizer",
    "InferencePipeline",
]