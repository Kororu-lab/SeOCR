import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from ocr_model.config.config import Config
from ocr_model.utils.data_utils import OCRDataset, collate_fn
from ocr_model.detector.model import TextDetector

def train_detector(config: Config, is_test: bool = False):
    # Initialize tensorboard
    writer = SummaryWriter(config.log_dir / "detector")
    
    # Set device
    device = torch.device(config.detector.device)
    
    # Create dataset and dataloader
    train_dataset = OCRDataset(
        Path(config.data.predata_dir),
        config,
        is_train=True,
        is_test=is_test
    )
    val_dataset = OCRDataset(
        Path(config.data.predata_dir),
        config,
        is_train=False,
        is_test=is_test
    )
    
    batch_size = config.test.test_batch_size if is_test else config.detector.batch_size
    num_epochs = config.test.test_epochs if is_test else config.detector.num_epochs
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    
    print(f"Test Mode: {is_test}")
    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Val Dataset Size: {len(val_dataset)}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    
    # Create model
    model = TextDetector(config.detector).to(device)
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.detector.learning_rate,
        weight_decay=config.detector.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=config.detector.learning_rate * 0.1
    )
    
    # Create loss functions
    region_criterion = nn.BCEWithLogitsLoss()
    affinity_criterion = nn.BCEWithLogitsLoss()
    detection_criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        train_region_loss = 0
        train_affinity_loss = 0
        train_detection_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            # Move data to device
            images = batch["images"].to(device)
            region_maps = batch["region_maps"].unsqueeze(1).to(device)
            affinity_maps = batch["affinity_maps"].unsqueeze(1).to(device)
            detection_maps = batch["detection_maps"].unsqueeze(1).to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate losses
            region_loss = region_criterion(outputs.craft_outputs["region_score"], region_maps)
            affinity_loss = affinity_criterion(outputs.craft_outputs["affinity_score"], affinity_maps)
            detection_loss = detection_criterion(outputs.craft_outputs["detection"], detection_maps)
            loss = region_loss + affinity_loss + detection_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_region_loss += region_loss.item()
            train_affinity_loss += affinity_loss.item()
            train_detection_loss += detection_loss.item()
            
            pbar.set_postfix({
                "loss": loss.item(),
                "region_loss": region_loss.item(),
                "affinity_loss": affinity_loss.item(),
                "detection_loss": detection_loss.item()
            })
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        train_region_loss /= len(train_loader)
        train_affinity_loss /= len(train_loader)
        train_detection_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        val_region_loss = 0
        val_affinity_loss = 0
        val_detection_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                images = batch["images"].to(device)
                region_maps = batch["region_maps"].unsqueeze(1).to(device)
                affinity_maps = batch["affinity_maps"].unsqueeze(1).to(device)
                detection_maps = batch["detection_maps"].unsqueeze(1).to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Calculate losses
                region_loss = region_criterion(outputs.craft_outputs["region_score"], region_maps)
                affinity_loss = affinity_criterion(outputs.craft_outputs["affinity_score"], affinity_maps)
                detection_loss = detection_criterion(outputs.craft_outputs["detection"], detection_maps)
                loss = region_loss + affinity_loss + detection_loss
                
                # Update metrics
                val_loss += loss.item()
                val_region_loss += region_loss.item()
                val_affinity_loss += affinity_loss.item()
                val_detection_loss += detection_loss.item()
        
        # Calculate average losses
        val_loss /= len(val_loader)
        val_region_loss /= len(val_loader)
        val_affinity_loss /= len(val_loader)
        val_detection_loss /= len(val_loader)
        
        # Log metrics to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Region_Loss/train', train_region_loss, epoch)
        writer.add_scalar('Region_Loss/val', val_region_loss, epoch)
        writer.add_scalar('Affinity_Loss/train', train_affinity_loss, epoch)
        writer.add_scalar('Affinity_Loss/val', val_affinity_loss, epoch)
        writer.add_scalar('Detection_Loss/train', train_detection_loss, epoch)
        writer.add_scalar('Detection_Loss/val', val_detection_loss, epoch)
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                config.checkpoint_dir / "detector_best.pth"
            )
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print("-" * 50)
    
    writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    args = parser.parse_args()
    
    config = Config()
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config_dict = json.load(f)
            # TODO: config_dict를 Config 객체로 변환
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    train_detector(config, is_test=args.test)

if __name__ == "__main__":
    main() 