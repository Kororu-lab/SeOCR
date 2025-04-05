import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class TextClassificationDataset(Dataset):
    def __init__(self, data_dir, config, is_train=True):
        self.data_dir = Path(data_dir)
        self.config = config
        self.is_train = is_train
        
        # 데이터 전처리
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # 데이터 로드
        self.samples = self._load_dataset()
        
    def _load_dataset(self):
        samples = []
        for json_file in self.data_dir.glob("**/*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 이미지 파일 경로
            img_path = json_file.with_suffix('.png')
            if not img_path.exists():
                continue
                
            # 텍스트 영역과 라벨 추출
            for text_coord in data['Text_Coord']:
                bbox = text_coord['Bbox']
                text = text_coord['annotate']
                
                samples.append({
                    'image_path': str(img_path),
                    'bbox': bbox,
                    'text': text
                })
                
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 이미지 로드 및 전처리
        img = cv2.imread(sample['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 텍스트 영역 추출
        x, y, w, h = sample['bbox'][:4]
        text_region = img[y:y+h, x:x+w]
        
        # 이미지 전처리
        text_region = Image.fromarray(text_region)
        text_region = self.transform(text_region)
        
        # 텍스트를 클래스 인덱스로 변환
        text = sample['text']
        label = self.config.classifier.char_to_idx[text]
        
        return {
            'image': text_region,
            'label': label,
            'text': text
        }

def create_dataloaders(config):
    train_dataset = TextClassificationDataset(
        config.data.predata_dir,
        config,
        is_train=True
    )
    
    val_dataset = TextClassificationDataset(
        config.data.predata_dir,
        config,
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.classifier.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.classifier.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    return train_loader, val_loader 