from transformers import CLIPProcessor
#!/usr/bin/env python
import argparse
from transformers.data import processors

parser = argparse.ArgumentParser()
parser.add_argument("model_dir", help="Path to directory containing config.json and flax_model.msgpack")

args = parser.parse_args()

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
processor.save_pretrained(args.model_dir)