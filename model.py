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
        # features = text_features * image_features # --> HateCLIPper
        output = self.mlp(features)
        return output


class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        attn_output, _ = self.multihead_attn(query, key_value, key_value)
        x = self.norm(query + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm_ffn(x + self.dropout(ffn_output))
        return x


class CrossAttentionClassifier(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.3):
        super().__init__()

        self.proj_text = nn.Sequential(
            nn.Linear(512, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )
        self.proj_image = nn.Sequential(
            nn.Linear(512, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )

        self.attn_text = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.attn_image = CrossModalAttention(hidden_dim, num_heads, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text, image):
        feat_text = self.proj_text(text).unsqueeze(1)
        feat_image = self.proj_image(image).unsqueeze(1)

        # Cross Attention
        feat_text_enhanced = self.attn_text(feat_text, feat_image)
        feat_image_enhanced = self.attn_image(feat_image, feat_text)

        # Squeeze & Concat
        feat_text_enhanced = feat_text_enhanced.squeeze(1)
        feat_image_enhanced = feat_image_enhanced.squeeze(1)

        combined = torch.cat([feat_text_enhanced, feat_image_enhanced], dim=1)

        output = self.classifier(combined)
        return output
