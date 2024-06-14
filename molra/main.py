import os
import io
import csv
import requests
import json
from molra.detection.dino import process_detections
from molra.classification.plantnet import PlantNet
from molra.utils import nms_custom 
from PIL import Image, ImageDraw
from uuid import uuid4
import torch
from torchvision.ops import nms
from transformers import pipeline

class_paths = {
    "fern": "./classes/balbina_ferns.csv",
    "flower": "./classes/balbina_flowers.csv",
    "palm": "./classes/balbina_palms.csv",
    "plant": "./classes/balbina_plants.csv"
}

def fetch_species(class_key: str, all: bool = False):
    # Map 'berry' to 'flower'
    if class_key == "berry":
        class_key = "flower"
    
    # Get the CSV file path from the class_paths dictionary
    classes_file = class_paths.get(class_key)
    if not classes_file:
        raise ValueError(f"Invalid class key: {class_key}")

    # Load the CSV file
    with open(classes_file, 'r') as f:
        csv_reader = csv.DictReader(f)
        if all:
            species_names = [row for row in csv_reader]
        else:
            species_names = [{'name': row['scientificname']} for row in csv_reader if row['scientificname']]
    
    return species_names

def save_cropped_image(cropped_image, label):
    os.makedirs("./output", exist_ok=True)
    cropped_image_path = f"./output/{label}_{uuid4()}.jpeg"
    cropped_image.save(cropped_image_path)

def convert_bbox_to_absolute(bbox, image_size):
    width, height = image_size
    cx = bbox[0] * width
    cy = bbox[1] * height
    w = bbox[2] * width
    h = bbox[3] * height

    left = int(cx - w / 2)
    top = int(cy - h / 2)
    right = int(cx + w / 2)
    bottom = int(cy + h / 2)

    return left, top, right, bottom

def crop_image(original_image, left, top, right, bottom):
    cropped_image = original_image.crop((left, top, right, bottom))
    return cropped_image

def plot_image(original_image, bounding_boxes, image_name):
    draw = ImageDraw.Draw(original_image)
    for box in bounding_boxes:
        label = box['label']
        left, top, right, bottom = box['absolute_bbox']

        draw.rectangle([left, top, right, bottom], outline="red", width=2)
        draw.text((left, top), label, fill="red")

    annotated_image_path = f"./output/{image_name}_annotated.jpeg"
    original_image.save(annotated_image_path)

def filter(bounding_boxes, image_height, image_width, iou_threshold=0.5, max_preds_per_image=10):
    # Filter 1: Remove boxes above 90% of both the width and height of the image
    initial_count = len(bounding_boxes)
    filtered_boxes = [
        box for box in bounding_boxes
        if box['bbox'][2] <= 0.9 and box['bbox'][3] <= 0.9
    ]

    # Filter 2: Remove boxes that are inside another box
    def is_inside(box1, box2):
        return (box1[0] >= box2[0] and box1[1] >= box2[1] and
                box1[0] + box1[2] <= box2[0] + box2[2] and
                box1[1] + box1[3] <= box2[1] + box2[3])

    non_inside_boxes = []
    for i, box1 in enumerate(filtered_boxes):
        inside = False
        for j, box2 in enumerate(filtered_boxes):
            if i != j and is_inside(box1['bbox'], box2['bbox']):
                inside = True
                break
        if not inside:
            non_inside_boxes.append(box1)

    # Filter 3: Apply custom NMS to combine similar boxes
    nms_boxes = nms_custom(non_inside_boxes, iou_threshold)

    # Filter 4: Filter the number of boxes by max_preds_per_image based on logit (i.e. confidence) value
    sorted_boxes = sorted(nms_boxes, key=lambda x: x['logit'], reverse=True)
    final_boxes = sorted_boxes[:max_preds_per_image]
    return final_boxes

def process_pipeline(image_path, text_prompt, box_threshold, text_threshold, plot_annotated_image=False, save_crops=False, classify_image=True, save_annotations=False, max_preds_per_image=5, nms_threshold=0.5):
    # Convert list of classes to a comma-separated string
    text_prompt = ", ".join(text_prompt)
    
    # Load the image from the file path
    original_image = Image.open(image_path).convert("RGB")
    with io.BytesIO() as output:
        original_image.save(output, format="JPEG")
        image_data = output.getvalue()
    
    bounding_boxes = process_detections(image_data, text_prompt, box_threshold, text_threshold)

    # Convert bounding boxes to absolute coordinates
    for box in bounding_boxes:
        bbox = box['bbox']
        box['absolute_bbox'] = convert_bbox_to_absolute(bbox, original_image.size)

    # Apply NMS after converting to absolute coordinates
    bounding_boxes = filter(bounding_boxes, original_image.height, original_image.width, nms_threshold, max_preds_per_image)

    results_list = []

    image_name = os.path.basename(image_path).split('.')[0]  

    if plot_annotated_image:
        plot_image(original_image, bounding_boxes, image_name)

    if classify_image:
        for box in bounding_boxes:
            label = box['label'].split()[0]
            left, top, right, bottom = box['absolute_bbox']
            cropped_image = crop_image(original_image, left, top, right, bottom)
            
            if save_crops:
                save_cropped_image(cropped_image, label)
            
            if label == 'fern':
                species_names = fetch_species(label, all=True)
                fern_classifier = pipeline("image-classification", model="ebayes/amazonas-fern-latest")
                classifier_output = fern_classifier(cropped_image)
                top_fern_results = sorted(classifier_output, key=lambda x: x['score'], reverse=True)[:3]

                processed_labels = []
                for res in top_fern_results:
                    label = res['label'].replace('_', ' ')
                    matching_species = next((species for species in species_names if species['name'] == label), None)
                    if matching_species:
                        processed_labels.append([label, matching_species['genus'], matching_species['family']])
                    else:
                        processed_labels.append([label, None, None]) 
                
                result_dict = {
                    "taxa": "fern",
                    "bbox": [left, top, right, bottom],
                    "confidence": [res['score'] for res in top_fern_results],
                    "labels": processed_labels
                }
                print(result_dict)
            else:
                species_names = fetch_species(label, all=False)
                plantnet = PlantNet(classes=species_names)
                
                # Identify the plant using PlantNet
                results = plantnet.identify(cropped_image)
                
                result_dict = {
                    "taxa": label,
                    "bbox": [left, top, right, bottom],
                    "confidence": [res['score'] for res in results],
                    "labels": [[res['scientificNameWithoutAuthor'], res['genus'], res['family'], res['commonNames']] for res in results]
                }
            
            results_list.append(result_dict)

    # Save annotations if required
    if save_annotations:
        with open(f"./output/{image_name}_annotations.json", "w") as f:
            json.dump(results_list, f, indent=4)

    # Return the results as a JSON object
    return json.dumps(results_list, indent=4)