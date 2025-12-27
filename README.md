# CyrillicHist-Processing-Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive pipeline for automatic symbol recognition in Cyrillic historical manuscripts (CyrillicHist). This project provides tools for detecting text corpora and segmenting lines in historical documents with subsequent annotation generation in JSON format for dataset publication on Zenodo (https://zenodo.org/records/18066472)

## Features

- **Text Block Detection**: Automatic detection of text blocks in historical manuscripts
- **Line Segmentation**: Precise segmentation of individual text lines within corpora
- **Adaptive Binarization**: Automatic image processing and binarization of text lines
- **Annotation Generation**: Creation of structured annotations in JSON format for dataset publication
- **YOLO Integration**: Utilizes state-of-the-art YOLO models for computer vision tasks
- **Scientifically Rigorous**: Designed for research publication in SCOPUS-indexed venues

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8 (recommended for GPU acceleration)
- 8GB+ RAM (16GB recommended for large manuscripts)

### Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Project Structure

```
historical_manuscript_processing/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── inference.py                 # Main inference script
├── config.py                    # Configuration settings
├── .gitignore                   # Git ignore file
└── src/                         # Source code
    ├── __init__.py
    ├── image_processor.py       # Image processing utilities
    ├── corpus_detector.py       # Text corpus detection
    ├── line_segmenter.py        # Line segmentation
    ├── segmentation_visualizer.py # Visualization tools
    └── inference_pipeline.py    # Main processing pipeline
```

## Quick Start

### Basic Usage

```bash
# Process a single manuscript image
python inference.py --image path/to/manuscript.jpg --output-dir results
```

### Expected Output

After running the pipeline, you'll get:

```
results/
├── manuscript_visualization.jpg     # Full visualization with detection & segmentation
├── manuscript_annotation.json       # Structured annotation in JSON format
├── detection/                       # Corpus detection results
│   ├── manuscript_visualization.jpg
│   └── manuscript_corpus_1.jpg
└── lines/                           # Individual segmented lines
    ├── manuscript_corpus_1_line_1.jpg
    ├── manuscript_corpus_1_line_2.jpg
    └── ...
```

## Advanced Usage

### Custom Models and Parameters

```bash
# Full pipeline with custom models and parameters
python inference.py \
    --image data/manuscripts/evangel_3.jpg \
    --detect-model models/v1/yolo_detect_corpus/detect_n_model/weights/best.pt \
    --segment-model models/v1/yolo_segment_lines/segment_n_model/weights/best.pt \
    --output-dir processed_results \
    --target-size 1280 \
    --no-save  # Skip visualization, only get JSON annotation
```

### Batch Processing

```bash
# Process multiple images (using a shell script)
for img in data/manuscripts/*.jpg; do
    python inference.py --image "$img" --output-dir "results/$(basename "$img" .jpg)"
done
```

## Python API Usage

### Basic Pipeline Integration

```python
from src import InferencePipeline

# Initialize pipeline with custom models
pipeline = InferencePipeline(
    detect_model_path="models/detect_corpus.pt",
    segment_model_path="models/segment_lines.pt",
    target_size=1280
)

# Run inference
results = pipeline.run("path/to/manuscript.jpg")

# Run with automatic saving
results = pipeline.run_and_save(
    img_path="path/to/manuscript.jpg",
    output_dir="results",
    base_filename="manuscript"
)
```

### Individual Component Usage

```python
from src import CorpusDetector, LineSegmenter, SegmentationVisualizer
import cv2

# Load image
image = cv2.imread("manuscript.jpg")

# Initialize detector
detector = CorpusDetector("models/detect_corpus.pt")
bboxes = detector.detect(image)

# Initialize segmenter
segmenter = LineSegmenter("models/segment_lines.pt")

# Process each corpus
for i, bbox in enumerate(bboxes):
    x1, y1, x2, y2 = map(int, bbox)
    corpus_img = image[y1:y2, x1:x2]
    lines = segmenter.segment_corpus(
        corpus_img, bbox, 
        image.shape[1], image.shape[0]
    )
    
    # Save individual lines
    for line in lines:
        SegmentationVisualizer.save_single_line(
            image, bbox, line["mask"],
            line["number_line"], i+1,
            "output_lines", "manuscript"
        )
```

## JSON Annotation Format

The pipeline generates structured annotations following a standardized format suitable for publication and dataset sharing:

```json
{
  "img_path": img_path,
  "detect_text_block": [
      {
          "number_text_block": int,
          "bbox": [x1, y1, x2, y2],
          "segment_lines": [
              {
                  "number_line": int,
                  "mask": [[x1,y1], [x2,y2], ...]
              }
          ]
      }
  ]
}
```

### Annotation Structure

- **img_path**: Relative path to the original image
- **detect_text_block**: Array of detected text corpora
  - `number_text_block`: Text block identifier
  - `bbox`: Bounding box [x1, y1, x2, y2] in absolute coordinates
  - `segment_lines`: Array of segmented lines within the corpus
    - `number_line`: Line identifier within corpus
    - `mask`: Polygon coordinates for line segmentation

### Examples Annotations

![Example of segmentation annotations of CyrillicHist dataset.](examples_annotations/examples.png)

## Pretrained Models

### Available Models

Download pretrained models and place them in the `models/` directory:

```bash
models/
├── v1/
│   ├── yolo_detect_corpus/
│   │   └── detect_n_model/
│   │       └── weights/
│   │           └── best.pt
│   └── yolo_segment_lines/
│       └── segment_n_model/
│           └── weights/
│               └── best.pt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 [Andrey Nikishin]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

## Contact

- **Author**: Andrey Nikishin
- **Email**: nikishin.ap.article@gmail.com
- **Institution**: ITMO University
- **GitHub**: [@NikishinAndrey](https://github.com/NikishinAndrey)
