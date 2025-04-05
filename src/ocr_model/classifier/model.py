import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
from ocr_model.config.config import Config

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = input + x
        return x

class ConvNeXt(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=4),
            nn.LayerNorm(64)
        )
        
        # Stages
        self.stages = nn.ModuleList([
            nn.Sequential(
                ConvNeXtBlock(64),
                ConvNeXtBlock(64),
                ConvNeXtBlock(64)
            ),
            nn.Sequential(
                nn.LayerNorm(64),
                nn.Conv2d(64, 128, kernel_size=2, stride=2),
                *[ConvNeXtBlock(128) for _ in range(3)]
            ),
            nn.Sequential(
                nn.LayerNorm(128),
                nn.Conv2d(128, 256, kernel_size=2, stride=2),
                *[ConvNeXtBlock(256) for _ in range(3)]
            ),
            nn.Sequential(
                nn.LayerNorm(256),
                nn.Conv2d(256, 512, kernel_size=2, stride=2),
                *[ConvNeXtBlock(512) for _ in range(3)]
            )
        ])
        
        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, config.classifier.transformer_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = x.mean([-2, -1])
        x = self.head(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.classifier.max_text_length, config.classifier.transformer_dim)
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.classifier.transformer_dim,
            nhead=config.classifier.transformer_heads,
            dim_feedforward=config.classifier.transformer_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.classifier.transformer_layers
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1)]
        
        # Transformer encoding
        x = self.transformer(x)
        
        return x

class TextClassifier(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # ConvNeXt backbone
        self.backbone = ConvNeXt(config)
        
        # Transformer encoder
        self.transformer = TransformerEncoder(config)
        
        # Classification head
        self.classifier = nn.Linear(
            config.classifier.transformer_dim,
            config.classifier.num_classes
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Backbone features
        features = self.backbone(x)
        
        # Reshape for transformer
        B, C = features.shape
        features = features.view(B, 1, C)
        
        # Transformer encoding
        encoded = self.transformer(features)
        
        # Classification
        logits = self.classifier(encoded)
        
        return {
            "logits": logits,
            "features": features
        } 