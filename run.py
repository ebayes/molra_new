import os
from molra.main import process_pipeline

input_dir = 'input'
CLASSES = ["flower", "berry", "plant", "fern", "palm"]
BOX_THRESHOLD = 0.2
TEXT_THRESHOLD = 0.2

for image_file in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_file)
    output = process_pipeline(
        image_path=image_path,
        text_prompt=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        # plot_annotated_image=True,
        # save_crops=True,
        # classify_image=True,
        # save_annotations=True,
        # max_preds_per_image=5,
        # nms_threshold=0.5
    )
    print(output)