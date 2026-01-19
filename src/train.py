#!/usr/bin/env python3
"""
train.py - Reproducible ResNet18 training pipeline for pneumonia detection
Extracted from Jupyter notebook for production reproducibility
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from deeplake import load
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
from tqdm import tqdm

def main():
    BATCH_SIZE = 32
    EPOCHS = 3
    LR = 1e-4
    
    ds = load("hub://activeloop/chest-xray-pneumonia") 
    print(f"Dataset loaded: {len(ds)} samples")
    
    
    train_idx, temp_idx = train_test_split(range(len(ds)), test_size=0.4, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    label_mapping = {0: 0, 1: 1, 2: 1}  
    transform = {
        "images": lambda x: torch.tensor(x).unsqueeze(0),  
        "labels": lambda y: torch.tensor(label_mapping[int(y.item())]).long()
    }
    
    train_loader = ds.pytorch(tensors=["images", "labels"], indices=train_idx, 
                             batch_size=BATCH_SIZE, transform=transform, shuffle=True)
    val_loader = ds.pytorch(tensors=["images", "labels"], indices=val_idx, 
                           batch_size=BATCH_SIZE, transform=transform, shuffle=False)
    
    # Production model: ResNet18 (grayscale)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x = batch["images"].to(device)
            y = batch["labels"].to(device).squeeze().long()
            
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} loss: {total_loss/len(train_loader):.4f}")
    
    
    Path("artifacts").mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": 2,
        "architecture": "resnet18"
    }, "artifacts/resnet18_leakfree.pth")
    
    print("PRODUCTION MODEL SAVED: artifacts/resnet18_leakfree.pth")

if __name__ == "__main__":
    main()
