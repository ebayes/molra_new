import numpy as np

def calculate_iou(box1, box2):
    # Calculate the (x, y)-coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(box1Area + box2Area - interArea)

    return iou

def nms_custom(bounding_boxes, iou_threshold):
    if len(bounding_boxes) == 0:
        return []

    # Initialize the list of picked indexes
    picked_boxes = []

    # Extract the bounding boxes and scores
    boxes = np.array([box['absolute_bbox'] for box in bounding_boxes])
    scores = np.array([box['logit'] for box in bounding_boxes])

    # Sort the bounding boxes by the score in descending order
    indices = np.argsort(scores)[::-1]

    while len(indices) > 0:
        # Pick the box with the highest score
        current = indices[0]
        picked_boxes.append(bounding_boxes[current])

        # Compute IoU of the picked box with the rest
        rest_indices = indices[1:]
        ious = np.array([calculate_iou(boxes[current], boxes[i]) for i in rest_indices])

        # Select boxes with IoU less than the threshold
        indices = rest_indices[ious < iou_threshold]

    return picked_boxes