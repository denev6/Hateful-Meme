from transformers import CLIPModel, CLIPProcessor
import os

model_id = "openai/clip-vit-base-patch32"
save_directory = "model/clip-vit-base-patch32"

os.makedirs(save_directory, exist_ok=True)

print(f"Downloading {model_id}...")

model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

model.save_pretrained(save_directory)
processor.save_pretrained(save_directory)

print(f"Model and processor saved to {save_directory}")
