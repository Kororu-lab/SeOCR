import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Dict, Any

from ocr_model.config.config import Config
from ocr_model.utils.data_utils import create_data_loaders
from ocr_model.detector.model import DBNetPP

def train_detector(config: Config, is_test: bool = False) -> None:
    """
    DBNet++ 모델 학습 함수
    
    Args:
        config: 설정 객체
        is_test: 테스트 모드 여부
    """
    # TensorBoard 설정
    writer = SummaryWriter(config.log_dir / "detector")
    
    # 데이터 로더 생성
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # 모델 초기화
    model = DBNetPP(config).to(config.detector.device)
    
    # 옵티마이저 설정
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.detector.learning_rate,
        weight_decay=config.detector.weight_decay
    )
    
    # 스케줄러 설정
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.detector.learning_rate,
        epochs=config.detector.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=config.detector.warmup_epochs / config.detector.num_epochs
    )
    
    # 손실 함수
    criterion = nn.BCEWithLogitsLoss()
    
    # 최적 모델 저장 경로
    best_model_path = config.checkpoint_dir / "detector_best.pth"
    
    # 학습 루프
    best_val_loss = float("inf")
    for epoch in range(config.detector.num_epochs):
        # 학습
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.detector.num_epochs} [Train]"):
            # 데이터 이동
            detector_images = batch["detector_images"].to(config.detector.device)
            region_maps = batch["region_maps"].to(config.detector.device)
            affinity_maps = batch["affinity_maps"].to(config.detector.device)
            detection_maps = batch["detection_maps"].to(config.detector.device)
            
            # 순전파
            outputs = model(detector_images)
            probability_map = outputs["probability_map"]
            threshold_map = outputs["threshold_map"]
            
            # 손실 계산
            prob_loss = criterion(probability_map, detection_maps)
            thresh_loss = criterion(threshold_map, detection_maps)
            loss = prob_loss + thresh_loss
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # 통계 업데이트
            train_loss += loss.item()
            train_steps += 1
            
            if is_test and train_steps >= 10:
                break
        
        # 검증
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config.detector.num_epochs} [Val]"):
                # 데이터 이동
                detector_images = batch["detector_images"].to(config.detector.device)
                region_maps = batch["region_maps"].to(config.detector.device)
                affinity_maps = batch["affinity_maps"].to(config.detector.device)
                detection_maps = batch["detection_maps"].to(config.detector.device)
                
                # 순전파
                outputs = model(detector_images)
                probability_map = outputs["probability_map"]
                threshold_map = outputs["threshold_map"]
                
                # 손실 계산
                prob_loss = criterion(probability_map, detection_maps)
                thresh_loss = criterion(threshold_map, detection_maps)
                loss = prob_loss + thresh_loss
                
                # 통계 업데이트
                val_loss += loss.item()
                val_steps += 1
                
                if is_test and val_steps >= 10:
                    break
        
        # 평균 손실 계산
        train_loss /= train_steps
        val_loss /= val_steps
        
        # TensorBoard 로깅
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)
        
        # 최적 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss
            }, best_model_path)
        
        print(f"Epoch {epoch + 1}/{config.detector.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 50)
    
    writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="테스트 모드 실행")
    args = parser.parse_args()
    
    config = Config()
    train_detector(config, is_test=args.test)

if __name__ == "__main__":
    main() 