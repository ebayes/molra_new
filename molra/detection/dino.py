import io
from PIL import Image
import numpy as np
import torch
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from typing import Tuple

def load_image_from_memory(image_data: bytes) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def process_detections(image_data, text_prompt, box_threshold, text_threshold):
    CONFIG_PATH = "molra/detection/dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    WEIGHTS_PATH = "molra/detection/dino/weights/groundingdino_swint_ogc.pth"    
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    image_source, image = load_image_from_memory(image_data)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device="cpu"
    )

    data = []
    for box, label, logit in zip(boxes, phrases, logits):
        data.append({
            "bbox": box.tolist(),  
            "label": label,
            "logit": logit
        })

    return data