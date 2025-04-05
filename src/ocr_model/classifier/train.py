import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from .model import TextClassifier
from .utils.data_utils import create_dataloaders
from ..config.config import Config

def train_classifier(config: Config, is_test: bool = False):
    # Initialize tensorboard
    writer = SummaryWriter(config.log_dir / "classifier")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Create model
    model = TextClassifier(config)
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.model.parameters(),
        lr=config.classifier.learning_rate,
        weight_decay=config.classifier.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.classifier.num_epochs,
        eta_min=config.classifier.learning_rate * 0.1
    )
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config.classifier.num_epochs):
        # Train
        model.model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.classifier.num_epochs}")
        for batch in pbar:
            # Move data to device
            images = batch['image'].to(model.device)
            labels = batch['label'].to(model.device)
            
            # Forward pass
            outputs = model.model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * train_correct / train_total
            })
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average metrics
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validate
        model.model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                images = batch['image'].to(model.device)
                labels = batch['label'].to(model.device)
                
                # Forward pass
                outputs = model.model(images)
                loss = criterion(outputs, labels)
                
                # Update metrics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate average metrics
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Log metrics to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.model.state_dict(),
                config.checkpoint_dir / "classifier_best.pth"
            )
        
        print(f"Epoch {epoch + 1}/{config.classifier.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print("-" * 50)
    
    writer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    config = Config()
    train_classifier(config, args.test) 