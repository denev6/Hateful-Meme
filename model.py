import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class CLIPFusionNetwork(nn.Module):
    def __init__(self, model_id="model/clip-vit-base-patch32"):
        super().__init__()

        self._clip = CLIPModel.from_pretrained(model_id, dtype=torch.float16)
        self._processor = CLIPProcessor.from_pretrained(model_id, use_fast=True)
        self.embed_dim = self._clip.config.projection_dim

        for param in self._clip.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, texts, images, device="cuda"):
        inputs = self._processor(
            text=texts,
            images=images,
            truncation=True,
            max_length=77,
            return_tensors="pt",
            padding=True,
            do_rescale=False,
            do_normalize=True,
        ).to(device)

        image_features = self._clip.get_image_features(pixel_values=inputs.pixel_values)
        text_features = self._clip.get_text_features(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
        )

        # Normalization
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        return text_features, image_features


class MLPClassifier(nn.Module):
    def __init__(self, hidden_dim, input_dim=1024, output_dim=1, dropout=0.3):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, text_features, image_features):
        features = torch.concat([text_features, image_features], dim=1)
        output = self.mlp(features)
        return output
