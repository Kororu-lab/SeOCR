import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
from ..config.config import Config

class FPN(nn.Module):
    def __init__(self, in_channels: Tuple[int, ...], out_channels: int):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channel in in_channels:
            self.lateral_convs.append(
                nn.Conv2d(in_channel, out_channels, 1)
            )
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            )
        
    def forward(self, features: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        laterals = [
            lateral_conv(feature)
            for lateral_conv, feature in zip(self.lateral_convs, features)
        ]
        
        # Top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode="nearest"
            )
        
        # Build outputs
        outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels)
        ]
        
        return tuple(outs)

class ASF(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, features: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Global average pooling
        global_features = [
            self.global_avg_pool(feature)
            for feature in features
        ]
        
        # Channel attention
        attention_weights = [
            self.sigmoid(self.conv(feature))
            for feature in global_features
        ]
        
        # Weighted sum
        weighted_features = [
            feature * weight
            for feature, weight in zip(features, attention_weights)
        ]
        
        # Sum all features
        fused_features = sum(weighted_features)
        
        return fused_features

class DetectionHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.probability_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, stride=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1)
        )
        
        self.threshold_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, stride=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probability = self.probability_head(x)
        threshold = self.threshold_head(x)
        return probability, threshold

class DBNetPP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Backbone
        if config.detector.backbone == "resnet50":
            self.backbone = torch.hub.load(
                'pytorch/vision:v0.10.0',
                'resnet50',
                pretrained=config.detector.pretrained
            )
            self.backbone = nn.Sequential(
                *list(self.backbone.children())[:-2]
            )
        
        # FPN
        self.fpn = FPN(
            config.detector.fpn_in_channels,
            config.detector.fpn_out_channels
        )
        
        # ASF
        self.asf = ASF(config.detector.asf_channels)
        
        # Detection Head
        self.detection_head = DetectionHead(config.detector.asf_channels)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Backbone
        features = self.backbone(x)
        
        # FPN
        fpn_features = self.fpn(features)
        
        # ASF
        fused_features = self.asf(fpn_features)
        
        # Detection
        probability_map, threshold_map = self.detection_head(fused_features)
        
        return {
            "probability_map": probability_map,
            "threshold_map": threshold_map
        } 