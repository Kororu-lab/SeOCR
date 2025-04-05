import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def load_image(image_path: str) -> np.ndarray:
    """이미지를 로드하고 전처리합니다."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_json(json_path: str) -> Dict[str, Any]:
    """JSON 파일을 로드합니다."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_region_map(image_shape: Tuple[int, int], text_coords: List[List[float]]) -> np.ndarray:
    """텍스트 영역 맵을 생성합니다."""
    region_map = np.zeros(image_shape[:2], dtype=np.float32)
    for coords in text_coords:
        x1, y1, x2, y2 = coords
        cv2.rectangle(region_map, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1)
    return region_map

def generate_affinity_map(image_shape: Tuple[int, int], text_coords: List[List[float]]) -> np.ndarray:
    """텍스트 영역 간의 관계 맵을 생성합니다."""
    affinity_map = np.zeros(image_shape[:2], dtype=np.float32)
    for i in range(len(text_coords) - 1):
        x1, y1, x2, y2 = text_coords[i]
        next_x1, next_y1, next_x2, next_y2 = text_coords[i + 1]
        
        # 현재 텍스트 영역과 다음 텍스트 영역 사이의 중간점을 연결
        center1 = (int((x1 + x2) // 2), int((y1 + y2) // 2))
        center2 = (int((next_x1 + next_x2) // 2), int((next_y1 + next_y2) // 2))
        cv2.line(affinity_map, center1, center2, 1, 2)
    return affinity_map

def preprocess_image(image: np.ndarray, augment: bool = True) -> torch.Tensor:
    """이미지를 전처리하고 텐서로 변환합니다."""
    # 이미지 크기 조정
    target_size = (256, 256)
    image = cv2.resize(image, target_size)
    
    if augment:
        # 데이터 증강
        if np.random.random() > 0.5:
            angle = np.random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((target_size[0] // 2, target_size[1] // 2), angle, 1.0)
            image = cv2.warpAffine(image, M, target_size)
        
        # 밝기와 대비 조정
        alpha = np.random.uniform(0.8, 1.2)
        beta = np.random.uniform(-10, 10)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # 정규화
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    
    # 텐서로 변환
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    return image

class OCRDataset(Dataset):
    def __init__(self, data_dir: Path, config: Any, is_train: bool = True, is_test: bool = False):
        self.data_dir = data_dir
        self.config = config
        self.is_train = is_train
        self.is_test = is_test
        
        # 데이터셋 로드
        self.data = self._load_dataset()
        if is_test:
            self.data = self.data[:10]  # 테스트 모드에서는 10개의 샘플만 사용
    
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """데이터셋을 로드합니다."""
        dataset = []
        
        # 각 출판 유형별로 데이터 로드
        for pub_type in self.config.data_source.publication_types:
            pub_dir = self.data_dir / pub_type
            if not pub_dir.exists():
                continue
            
            # JSON 파일 찾기
            for json_file in pub_dir.glob("**/*.json"):
                data = load_json(str(json_file))
                image_path = str(json_file).replace(".json", ".png")
                
                if os.path.exists(image_path):
                    # 텍스트 좌표 추출
                    text_coords = []
                    for coord in data["Text_Coord"]:
                        x, y, w, h = coord["Bbox"][:4]
                        text_coords.append([x, y, x + w, y + h])
                    
                    dataset.append({
                        "image_path": image_path,
                        "text_coords": text_coords,
                        "text": [coord["annotate"] for coord in data["Text_Coord"]]
                    })
        
        print(f"Found {len(dataset)} samples in {self.data_dir}")
        
        # 테스트 모드에서는 10개의 샘플만 사용
        if self.is_test:
            dataset = dataset[:10]
        # 학습/검증 데이터 분할
        else:
            total = len(dataset)
            train_size = int(total * 0.8)  # 80%는 학습 데이터
            if self.is_train:
                dataset = dataset[:train_size]
            else:
                dataset = dataset[train_size:]
        
        return dataset
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """데이터셋에서 하나의 샘플을 가져옵니다."""
        item = self.data[idx]
        
        # 이미지 로드 및 전처리
        image = load_image(item["image_path"])
        original_size = image.shape[:2]
        image = preprocess_image(image, augment=self.is_train)
        
        # 텍스트 좌표 정규화
        text_coords = []
        for coords in item["text_coords"]:
            x1, y1, x2, y2 = coords
            x1 = x1 * 256 / original_size[1]
            y1 = y1 * 256 / original_size[0]
            x2 = x2 * 256 / original_size[1]
            y2 = y2 * 256 / original_size[0]
            text_coords.append([x1, y1, x2, y2])
        
        # 영역 맵과 관계 맵 생성
        region_map = generate_region_map((256, 256), text_coords)
        affinity_map = generate_affinity_map((256, 256), text_coords)
        detection_map = np.maximum(region_map, affinity_map)
        
        # 텐서로 변환
        region_map = torch.from_numpy(region_map).float()
        affinity_map = torch.from_numpy(affinity_map).float()
        detection_map = torch.from_numpy(detection_map).float()
        
        return {
            "images": image,
            "region_maps": region_map,
            "affinity_maps": affinity_map,
            "detection_maps": detection_map,
            "text": item["text"]
        }

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """배치 데이터를 처리합니다."""
    images = torch.stack([item["images"] for item in batch])
    region_maps = torch.stack([item["region_maps"] for item in batch])
    affinity_maps = torch.stack([item["affinity_maps"] for item in batch])
    detection_maps = torch.stack([item["detection_maps"] for item in batch])
    texts = [item["text"] for item in batch]
    
    return {
        "images": images,
        "region_maps": region_maps,
        "affinity_maps": affinity_maps,
        "detection_maps": detection_maps,
        "texts": texts
    } 