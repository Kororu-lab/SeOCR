import json
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset, DataLoader
from ..config.config import Config

def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> torch.Tensor:
    """
    이미지 전처리 함수
    
    Args:
        image: 입력 이미지 (H, W, C)
        target_size: 목표 크기 (H, W)
        mean: 평균값
        std: 표준편차
    
    Returns:
        전처리된 이미지 텐서
    """
    # 리사이징
    image = cv2.resize(image, target_size[::-1], interpolation=cv2.INTER_AREA)
    
    # 정규화
    image = image.astype(np.float32) / 255.0
    image = (image - np.array(mean)) / np.array(std)
    
    # 텐서 변환
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    
    return image

def generate_region_map(
    image_size: Tuple[int, int],
    text_coords: List[List[int]]
) -> np.ndarray:
    """
    텍스트 영역 맵 생성 함수
    
    Args:
        image_size: 이미지 크기 (H, W)
        text_coords: 텍스트 좌표 리스트 [[x1, y1, x2, y2], ...]
    
    Returns:
        텍스트 영역 맵
    """
    region_map = np.zeros(image_size, dtype=np.float32)
    
    for coords in text_coords:
        x1, y1, x2, y2 = coords
        region_map[y1:y2, x1:x2] = 1.0
    
    return region_map

def generate_affinity_map(
    image_size: Tuple[int, int],
    text_coords: List[List[int]]
) -> np.ndarray:
    """
    텍스트 어피니티 맵 생성 함수
    
    Args:
        image_size: 이미지 크기 (H, W)
        text_coords: 텍스트 좌표 리스트 [[x1, y1, x2, y2], ...]
    
    Returns:
        텍스트 어피니티 맵
    """
    affinity_map = np.zeros(image_size, dtype=np.float32)
    
    for i in range(len(text_coords) - 1):
        curr = text_coords[i]
        next = text_coords[i + 1]
        
        # 현재 박스와 다음 박스 사이의 중간 영역
        x1 = min(curr[2], next[0])
        x2 = max(curr[2], next[0])
        y1 = min(curr[1], next[1], curr[3], next[3])
        y2 = max(curr[1], next[1], curr[3], next[3])
        
        affinity_map[y1:y2, x1:x2] = 1.0
    
    return affinity_map

class OCRDataset(Dataset):
    def __init__(self, config: Config, mode: str = "train"):
        self.config = config
        self.mode = mode
        self.data = self._load_dataset()
        
    def _load_dataset(self) -> List[Dict]:
        data = []
        predata_dir = Path(self.config.data.predata_dir)
        
        # 출판물 형식별 디렉토리 처리
        for pub_format in ["필사본", "활자본", "목판본"]:
            format_dir = predata_dir / pub_format
            if not format_dir.exists():
                continue
                
            for json_file in format_dir.glob("*.json"):
                with open(json_file, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                
                # 이미지 파일 경로 생성
                image_path = json_file.with_suffix(".png")
                if not image_path.exists():
                    continue
                
                # 텍스트 좌표 정보 처리
                text_coords = []
                for coord in json_data.get("Text_Coord", []):
                    bbox = coord["Bbox"]
                    annotate = coord["annotate"]
                    
                    # 좌표 정규화
                    norm_coords = [
                        int(bbox[0] * self.config.data.detector_image_size[1]),
                        int(bbox[1] * self.config.data.detector_image_size[0]),
                        int(bbox[2] * self.config.data.detector_image_size[1]),
                        int(bbox[3] * self.config.data.detector_image_size[0])
                    ]
                    
                    text_coords.append({
                        "bbox": norm_coords,
                        "text": annotate
                    })
                
                data.append({
                    "image_path": str(image_path),
                    "text_coords": text_coords,
                    "metadata": {
                        "format": json_data["Publication_format"],
                        "category": json_data["Publication_category"],
                        "name": json_data["Publication_name"]
                    }
                })
        
        return data
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # 이미지 로드 및 전처리
        image = cv2.imread(item["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detector용 이미지
        detector_image = preprocess_image(
            image,
            self.config.data.detector_image_size
        )
        
        # 텍스트 좌표를 이용하여 region map과 affinity map 생성
        text_coords = [coord["bbox"] for coord in item["text_coords"]]
        region_map = generate_region_map(
            self.config.data.detector_image_size,
            text_coords
        )
        affinity_map = generate_affinity_map(
            self.config.data.detector_image_size,
            text_coords
        )
        detection_map = np.maximum(region_map, affinity_map)
        
        # Classifier용 이미지 패치
        text_patches = []
        text_labels = []
        for coord in item["text_coords"]:
            x1, y1, x2, y2 = coord["bbox"]
            patch = image[y1:y2, x1:x2]
            
            if patch.size == 0:
                continue
                
            patch = preprocess_image(
                patch,
                self.config.data.classifier_image_size
            )
            text_patches.append(patch)
            text_labels.append(coord["text"])
        
        return {
            "detector_image": detector_image,
            "text_patches": torch.stack(text_patches) if text_patches else torch.zeros(0, 3, *self.config.data.classifier_image_size),
            "text_labels": text_labels,
            "metadata": item["metadata"],
            "region_map": torch.from_numpy(region_map).float(),
            "affinity_map": torch.from_numpy(affinity_map).float(),
            "detection_map": torch.from_numpy(detection_map).float()
        }

def collate_fn(batch: List[Dict]) -> Dict:
    """
    배치 데이터를 처리하는 함수
    
    Args:
        batch: 배치 데이터 리스트
    
    Returns:
        처리된 배치 데이터
    """
    detector_images = torch.stack([item["detector_image"] for item in batch])
    region_maps = torch.stack([item["region_map"] for item in batch])
    affinity_maps = torch.stack([item["affinity_map"] for item in batch])
    detection_maps = torch.stack([item["detection_map"] for item in batch])
    
    # 텍스트 패치와 라벨 처리
    max_patches = max(len(item["text_patches"]) for item in batch)
    text_patches = []
    text_labels = []
    
    for item in batch:
        patches = item["text_patches"]
        labels = item["text_labels"]
        
        # 패딩 추가
        if len(patches) < max_patches:
            pad_size = max_patches - len(patches)
            patches = torch.cat([
                patches,
                torch.zeros(pad_size, *patches.shape[1:])
            ])
            labels.extend([""] * pad_size)
        
        text_patches.append(patches)
        text_labels.append(labels)
    
    text_patches = torch.stack(text_patches)
    
    return {
        "detector_images": detector_images,
        "region_maps": region_maps,
        "affinity_maps": affinity_maps,
        "detection_maps": detection_maps,
        "text_patches": text_patches,
        "text_labels": text_labels,
        "metadata": [item["metadata"] for item in batch]
    }

def create_data_loaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    데이터 로더 생성 함수
    
    Args:
        config: 설정 객체
    
    Returns:
        학습, 검증, 테스트 데이터 로더
    """
    # 전체 데이터셋 로드
    full_dataset = OCRDataset(config)
    
    # 데이터 분할
    dataset_size = len(full_dataset)
    train_size = int(config.data.train_ratio * dataset_size)
    val_size = int(config.data.val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader 