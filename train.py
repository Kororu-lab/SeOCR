import os
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from src import Config, SeOCRDataset, SeOCR

def main():
    # Create checkpoint directory
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    # Initialize model
    seocr = SeOCR()
    
    # Create dataset
    dataset = SeOCRDataset(Config.DATA_ROOT, seocr.processor)
    
    # Split dataset
    train_size = int(Config.TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS
    )
    
    # Train model
    seocr.train(train_dataloader, val_dataloader)

if __name__ == '__main__':
    main() 