from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import torch
import numpy as np
import random
import os

@dataclass
class DetectorConfig:
    # 모델 설정
    backbone: str = "resnet50"  # DBNet++ backbone
    pretrained: bool = True
    
    # FPN 설정
    fpn_in_channels: List[int] = (256, 512, 1024, 2048)
    fpn_out_channels: int = 256
    
    # ASF 설정
    asf_channels: int = 256
    
    # 학습 설정
    batch_size: int = 4
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    image_size: Tuple[int, int] = (1280, 1280)
    min_text_size: int = 10
    max_text_size: int = 1000
    shrink_ratio: float = 0.4
    thresh_min: float = 0.3
    thresh_max: float = 0.7

@dataclass
class ClassifierConfig:
    # 모델 설정
    backbone: str = "convnext_base"  # ConvNeXt backbone
    pretrained: bool = True
    transformer_dim: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 6
    
    # 학습 설정
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    image_size: Tuple[int, int] = (256, 256)
    max_text_length: int = 25
    num_classes: int = 2350  # 한글 자모 + 특수문자

@dataclass
class DataConfig:
    # 데이터 경로
    data_dir: Path = Path("data")
    predata_dir: str = "predata"
    
    # 데이터 전처리
    image_size: Tuple[int, int] = (256, 256)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # 데이터 증강
    augment: bool = True
    rotation_range: int = 5
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    detector_image_size: Tuple[int, int] = (1280, 1280)
    classifier_image_size: Tuple[int, int] = (256, 256)
    batch_size: int = 4
    num_workers: int = 4

@dataclass
class TestConfig:
    test_batch_size: int = 1
    test_epochs: int = 2
    test_image_size: Tuple[int, int] = (1280, 1280)

@dataclass
class Config:
    # 시드 설정
    seed: int = 42
    
    # 디렉토리 설정
    data_dir: Path = Path("data")
    predata_dir: str = "predata"
    log_dir: Path = Path("logs")
    checkpoint_dir: Path = Path("checkpoints")
    
    # 설정 객체
    detector: DetectorConfig = DetectorConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    data: DataConfig = DataConfig()
    test: TestConfig = TestConfig()
    
    # 기타 설정
    num_workers: int = 4
    
    def __post_init__(self):
        # 디렉토리 생성
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 시드 설정
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        # 환경 변수 설정
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 