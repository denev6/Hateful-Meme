from transformers import CLIPModel, CLIPProcessor
import os

# "clip-vit-base-patch32" or "clip-vit-large-patch14-336"
model_id = "openai/clip-vit-large-patch14-336"
save_directory = "model/clip-vit-large-patch14-336"

os.makedirs(save_directory, exist_ok=True)

print(f"Downloading {model_id}...")

model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

model.save_pretrained(save_directory)
processor.save_pretrained(save_directory)

print(f"Model and processor saved to {save_directory}")
