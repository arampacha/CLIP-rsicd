import argparse
import jax
import jax.numpy as jnp
import json
import matplotlib.pyplot as plt
import numpy as np
import requests
import os

from PIL import Image
from transformers import CLIPProcessor, FlaxCLIPModel


def encode_image(image_file, model, processor):
    image = Image.fromarray(plt.imread(os.path.join(IMAGES_DIR, image_file)))
    inputs = processor(images=image, return_tensors="jax")
    image_vec = model.get_image_features(**inputs)
    return np.array(image_vec).reshape(-1)


DATA_DIR = "/home/shared/data"
IMAGES_DIR = os.path.join(DATA_DIR, "rsicd_images")
CAPTIONS_FILE = os.path.join(DATA_DIR, "dataset_rsicd.json")
VECTORS_DIR = os.path.join(DATA_DIR, "vectors")
BASELINE_MODEL = "openai/clip-vit-base-patch32"

parser = argparse.ArgumentParser()
parser.add_argument("model_dir", help="Path to model to use for encoding")
args = parser.parse_args()

print("Loading image list...", end="")
image2captions = {}
with open(CAPTIONS_FILE, "r") as fcap:
    data = json.loads(fcap.read())
for image in data["images"]:
    if image["split"] == "test":
        filename = image["filename"]
        sentences = []
        for sentence in image["sentences"]:
            sentences.append(sentence["raw"])
        image2captions[filename] = sentences
    
print("{:d} images".format(len(image2captions)))


print("Loading model...")
if args.model_dir == "baseline":
    model = FlaxCLIPModel.from_pretrained(BASELINE_MODEL)
else:
    model = FlaxCLIPModel.from_pretrained(args.model_dir)
processor = CLIPProcessor.from_pretrained(BASELINE_MODEL)


model_basename = "-".join(args.model_dir.split("/")[-2:])
vector_file = os.path.join(VECTORS_DIR, "test-{:s}.tsv".format(model_basename))
print("Vectors written to {:s}".format(vector_file))
num_written = 0
fvec = open(vector_file, "w")
for image_file in image2captions.keys():
    if num_written % 100 == 0:
        print("{:d} images processed".format(num_written))
    image_vec = encode_image(image_file, model, processor)
    image_vec_s = ",".join(["{:.7e}".format(x) for x in image_vec])
    fvec.write("{:s}\t{:s}\n".format(image_file, image_vec_s))
    num_written += 1
    
print("{:d} images processed, COMPLETE".format(num_written))
fvec.close()

