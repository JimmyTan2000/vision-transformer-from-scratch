import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from model import VisionTransformer

def get_dataloaders(batch_size: int):
    # Standard CIFAR-10 transforms with augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return trainloader, testloader

def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Runs the forward pass with autocasting (mixed precision)
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Scales loss and calls backward to create scaled gradients
        scaler.scale(loss).backward()
        
        # Unscales the gradients and calls optimizer.step()
        scaler.step(optimizer)
        
        # Updates the scale for next iteration
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})

    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})

    return running_loss / len(dataloader), 100. * correct / total

def main():
    # Hyperparameters
    batch_size = 256
    epochs = 100
    learning_rate = 3e-4
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    print("Preparing dataloaders...")
    trainloader, testloader = get_dataloaders(batch_size)

    # Initialize Model
    # Using defaults: image_size=32, patch_size=4, hidden_dim=192, num_layers=12, num_heads=3
    print("Initializing Vision Transformer...")
    model = VisionTransformer().to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # Loss and Optimizer
    # betas=(0.9, 0.999) matches the Adam settings in the paper (Appendix B)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    )

    # Scheduler: Linear warmup followed by cosine decay (Appendix B of the paper).
    # Warmup lets the optimizer stabilize before taking large steps.
    warmup_epochs = 10  # ramp LR from ~0 -> learning_rate over first 10 epochs
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    
    # AMP Scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    os.makedirs('checkpoints', exist_ok=True)

    print("Starting training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, testloader, criterion, device)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if val_acc > best_acc:
            print(f"Validation accuracy improved from {best_acc:.2f}% to {val_acc:.2f}%. Saving model...")
            best_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/vit_best.pth')

if __name__ == "__main__":
    main()
