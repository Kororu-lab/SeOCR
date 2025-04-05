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
from ocr_model.classifier.model import TextClassifier

def train_classifier(config: Config, is_test: bool = False) -> None:
    """
    ConvNeXt + Transformer 모델 학습 함수
    
    Args:
        config: 설정 객체
        is_test: 테스트 모드 여부
    """
    # TensorBoard 설정
    writer = SummaryWriter(config.log_dir / "classifier")
    
    # 데이터 로더 생성
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # 모델 초기화
    model = TextClassifier(config).to(config.classifier.device)
    
    # 옵티마이저 설정
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.classifier.learning_rate,
        weight_decay=config.classifier.weight_decay
    )
    
    # 스케줄러 설정
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.classifier.learning_rate,
        epochs=config.classifier.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=config.classifier.warmup_epochs / config.classifier.num_epochs
    )
    
    # 손실 함수
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 패딩 토큰 무시
    
    # 최적 모델 저장 경로
    best_model_path = config.checkpoint_dir / "classifier_best.pth"
    
    # 학습 루프
    best_val_loss = float("inf")
    for epoch in range(config.classifier.num_epochs):
        # 학습
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.classifier.num_epochs} [Train]"):
            # 데이터 이동
            text_patches = batch["text_patches"].to(config.classifier.device)
            text_labels = batch["text_labels"]
            
            # 순전파
            outputs = model(text_patches)
            logits = outputs["logits"]
            
            # 손실 계산
            loss = criterion(logits.view(-1, config.classifier.num_classes), text_labels.view(-1))
            
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
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config.classifier.num_epochs} [Val]"):
                # 데이터 이동
                text_patches = batch["text_patches"].to(config.classifier.device)
                text_labels = batch["text_labels"]
                
                # 순전파
                outputs = model(text_patches)
                logits = outputs["logits"]
                
                # 손실 계산
                loss = criterion(logits.view(-1, config.classifier.num_classes), text_labels.view(-1))
                
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
        
        print(f"Epoch {epoch + 1}/{config.classifier.num_epochs}")
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
    train_classifier(config, is_test=args.test)

if __name__ == "__main__":
    main() 