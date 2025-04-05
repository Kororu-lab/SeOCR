from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
from pathlib import Path
import torch

@dataclass
class DataSourceConfig:
    use_predata: bool = True
    use_data: bool = False
    predata_path: str = "predata"
    data_path: str = "data"
    publication_types: List[str] = field(default_factory=lambda: ["필사본", "활자본", "목판본"])
    image_extensions: List[str] = field(default_factory=lambda: [".png", ".jpg", ".jpeg"])
    label_extensions: List[str] = field(default_factory=lambda: [".json"])

@dataclass
class DetectorConfig:
    # 모델 설정
    craft_pretrained: str = None
    
    # Transformer 설정
    transformer_num_layers: int = 1
    transformer_num_heads: int = 4
    transformer_hidden_dim: int = 64
    transformer_ff_dim: int = 128
    
    # 학습 설정
    batch_size: int = 1
    num_epochs: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class DataConfig:
    # 데이터 경로
    data_dir: Path = Path("data")
    predata_dir: Path = Path("predata")
    
    # 데이터 전처리
    image_size: Tuple[int, int] = (256, 256)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # 데이터 증강
    augment: bool = True
    rotation_range: int = 5
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)

@dataclass
class TestConfig:
    num_test_samples: int = 10  # 테스트용 샘플 수
    test_batch_size: int = 1    # 테스트용 배치 크기
    test_epochs: int = 2        # 테스트용 에포크 수

@dataclass
class Config:
    detector: DetectorConfig = DetectorConfig()
    data: DataConfig = DataConfig()
    test: TestConfig = TestConfig()
    data_source: DataSourceConfig = DataSourceConfig()
    
    # 공통 설정
    device: str = "cuda"  # 또는 "cpu"
    num_workers: int = 4
    seed: int = 42
    log_dir: Path = Path("logs")
    checkpoint_dir: Path = Path("checkpoints") 