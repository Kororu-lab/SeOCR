import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config.config import DetectorConfig

class CRAFT(nn.Module):
    def __init__(self, pretrained: str = None):
        super().__init__()
        
        # VGG16 기반의 백본
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # FPN
        self.fpn = nn.ModuleDict({
            'p5': nn.Conv2d(512, 256, 1),
            'p4': nn.Conv2d(512, 256, 1),
            'p3': nn.Conv2d(256, 256, 1),
            'p2': nn.Conv2d(128, 256, 1)
        })
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )
        
        # Region score head
        self.region_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )
        
        # Affinity score head
        self.affinity_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Backbone features
        c2 = self.backbone[:10](x)
        c3 = self.backbone[10:17](c2)
        c4 = self.backbone[17:24](c3)
        c5 = self.backbone[24:](c4)
        
        # FPN features
        p5 = self.fpn['p5'](c5)
        p4 = self.fpn['p4'](c4) + F.interpolate(p5, size=c4.shape[2:], mode='nearest')
        p3 = self.fpn['p3'](c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest')
        p2 = self.fpn['p2'](c2) + F.interpolate(p3, size=c2.shape[2:], mode='nearest')
        
        # Final feature map
        features = F.interpolate(p2, size=x.shape[2:], mode='nearest')
        
        # Detection outputs
        detection = self.detection_head(features)
        region_score = self.region_head(features)
        affinity_score = self.affinity_head(features)
        
        return {
            'detection': detection,
            'region_score': region_score,
            'affinity_score': affinity_score,
            'features': features
        }

class TextDetector(nn.Module):
    def __init__(self, config: DetectorConfig):
        super().__init__()
        
        # CRAFT 모델 초기화
        self.craft = CRAFT(config.craft_pretrained)
        
        # 특징 차원 변환
        self.feature_proj = nn.Conv2d(256, config.transformer_hidden_dim, 1)
        
        # Transformer 레이어
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.transformer_hidden_dim,
            nhead=config.transformer_num_heads,
            dim_feedforward=config.transformer_ff_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_num_layers
        )
        
        # 출력 레이어
        self.score_head = nn.Linear(config.transformer_hidden_dim, 1)
        self.box_head = nn.Linear(config.transformer_hidden_dim, 4)
    
    def forward(self, images):
        # CRAFT forward pass
        craft_outputs = self.craft(images)
        
        # Get features and project to transformer dimension
        features = craft_outputs["features"]
        features = self.feature_proj(features)
        
        # Transformer encoding
        B, C, H, W = features.shape
        features = features.view(B, C, -1).permute(0, 2, 1)
        features = self.transformer(features)
        
        # Prediction heads
        scores = self.score_head(features)
        boxes = self.box_head(features)
        
        return type('Outputs', (), {
            'logits': scores.squeeze(-1),
            'pred_boxes': boxes.view(B, -1, 4),
            'craft_outputs': craft_outputs
        }) 