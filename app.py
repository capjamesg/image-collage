import math
import os

import cv2
import numpy as np
import supervision as sv
import tqdm
import warnings

# suppressing warnings from pytorch and torch.utils
# that come from autodistill
warnings.filterwarnings("ignore")

from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--image_dir", type=str, help="Image directory", default="images")
parser.add_argument("--prompt", type=str, help="Prompt")
args = parser.parse_args()

prompt = args.prompt
image_dir = args.image_dir

if not prompt:
    exit("Please provide a prompt")

IMAGE_DIR = image_dir
IMAGE_H_W = (224, 224)

base_model = GroundingDINO(ontology=CaptionOntology({prompt: prompt}))

images = []

for image in tqdm.tqdm(os.listdir(IMAGE_DIR)):
    img = cv2.imread(os.path.join(IMAGE_DIR, image))
    result = base_model.predict(img)
    for r in result:
        # r[0] is the xyxy coordinates of the bounding box
        # corresponding to an image
        cropped_img = sv.crop_image(img, xyxy=r[0])
        images.append(cropped_img)

N = math.ceil(math.sqrt(len(images)))
images = [cv2.resize(img, IMAGE_H_W) for img in images]

height = IMAGE_H_W[0]
width = IMAGE_H_W[1]

image_count = len(images)
columns = math.ceil(math.sqrt(image_count))
rows = math.ceil(image_count / columns)

grid = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)

for idx, img in enumerate(images):
    x = idx % N
    y = idx // N

    grid[y * height : (y + 1) * height, x * width : (x + 1) * width] = img

cv2.imwrite("grid.png", grid)

print("Grid image saved as grid.png ✨")
