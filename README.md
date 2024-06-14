
# Oblique Drone Imagery Image Classification Pipeline

## Overview
This project provides a pipeline for detecting and classifying plant species in images in oblique drone imagery.

The pipeline processes images, detects bounding boxes around objects of interest, and classifies the detected objects into predefined classes.

## Features
- **Object Detection**: Detects objects in images using Grounding DINO and specified classes.
- **Plant Classification**: Classifies detected objects using the PlantNet API, filtered by species lists for the site.
- **Fern Classification**: Special handling for fern classification using a custom model.
- **[OPTIONAL] Non-Maximum Suppression (NMS)**: Filters overlapping bounding boxes.
- **[OPTIONAL] Image Cropping**: Crops detected objects from images.
- **[OPTIONAL] Annotation**: Annotates images with bounding boxes and labels.

## Requirements
- Python 3.7+
- Required Python packages (install via `requirements.txt`):
  - requests
  - Pillow
  - torch
  - transformers
  - numpy
  - python-dotenv

## Setup
1. Clone the repository:
   ```sh
   git clone <repository_url>
   ```

2. Install Grounding DINO and repo dependencies:
   ```sh
  cd molra/detection/dino
  pip install -e .
  cd ../../..
  pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env.local` file in the root directory.
   - Add your PlantNet API key:
     ```env
     PLANTNET_API_KEY=your_api_key_here
     ```

4. Prepare input images:
   - Place your input images in the `input` directory.

## Usage

### Running the Pipeline
To run the pipeline, execute the `run.py` script:
```sh
python run.py
```

### Configuration
You can configure the pipeline by modifying the following parameters in `run.py`:
- `input_dir`: Directory containing input images.
- `CLASSES`: List of classes to detect and classify.
- `BOX_THRESHOLD`: Threshold for bounding box detection (defaults to 0.2).
- `TEXT_THRESHOLD`: Threshold for text detection (defaults to 0.2).

## Output
The pipeline generates the following outputs:
- **Console Output**: Classification results printed to the console.
- **Annotated Images**: Images with bounding boxes and labels saved in the `output` directory (if `plot_annotated_image` is enabled).
- **Cropped Images**: Cropped images of detected objects saved in the `output` directory (if `save_crops` is enabled).
- **JSON Annotations**: JSON files containing classification results saved in the `output` directory (if `save_annotations` is enabled).

## File Structure
- `run.py`: Main script to run the pipeline.
- `molra/main.py`: Core pipeline logic.
- `molra/classification/plantnet.py`: PlantNet API and Hugging Face integration for plant and fern classification.
- `molra/detection/dino.py`: Object detection logic.
- `molra/utils.py`: Utility functions including NMS.
- `input/`: Directory for input images.
- `output/`: Directory for output images and annotations.
- `.env.local`: Environment variables file (not included).

## Example Output
The output JSON structure for each image is as follows:
```json
[
    {
      "taxa": class,
      "bbox": [
        x_min,
        y_min,
        x_max,
        y_max
      ],
      "confidence": [
        confidence_score_1,
        ...
      ],
      "labels": [
        [
          top_species_pred_1,
          top_genus_pred_1,
          top_family_pred_1
        ],
        ...
      ]
    }
]
```