import torch
import torchvision
import torchvision.transforms as transforms
from model import VisionTransformer
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

def load_model(checkpoint_path, device):
    # Initialize model with the same parameters as used in training
    model = VisionTransformer(
        image_size=32,
        patch_size=4,
        hidden_dim=192,
        num_layers=12,
        num_heads=3,
        mlp_dim=768,
        num_classes=10
    )
    
    # Load state dict
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Checkpoint {checkpoint_path} not found. Running with random weights.")
    
    model.to(device)
    model.eval()
    return model

def predict_sample(model, device):
    # Classes for CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Transform for validation/inference
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load a few test images
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True)
    
    # Get one batch
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    # Inference
    images_device = images.to(device)
    with torch.no_grad():
        outputs = model(images_device)
        _, predicted = torch.max(outputs, 1)
    
    # Show results
    print("Inference Results:")
    for i in range(4):
        print(f"  Sample {i+1}: GroundTruth: {classes[labels[i]]} | Predicted: {classes[predicted[i]]}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    checkpoint_path = "checkpoints/vit_best.pth"
    model = load_model(checkpoint_path, device)
    
    predict_sample(model, device)
