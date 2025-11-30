#!/usr/bin/env python3
"""
training/train.py
Transfer learning (ResNet-18) training script.
Designed for macOS M1/M2 (uses MPS if available) or CPU fallback.
Saves best model -> ../model/model.pth and ../model/labels.txt
"""
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18
import matplotlib.pyplot as plt

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='../dataset/dataset-resized/')
    p.add_argument('--output_dir', type=str, default='../model/')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--val_split', type=float, default=0.2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()

def main():
    args = parse_args()
    device = get_device()
    print("Using device:", device)

    os.makedirs(args.output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    class_names = dataset.classes
    num_classes = len(class_names)
    print("Found classes:", class_names)
    print("Total images:", len(dataset))

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Load pre-trained ResNet18
    model = resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    train_losses = []
    val_accuracies = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (train)"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} (val)"):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{args.epochs}  TrainLoss: {epoch_loss:.4f}  ValAcc: {val_acc*100:.2f}%")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.output_dir, "model.pth")
            torch.save(model.state_dict(), save_path)
            labels_path = os.path.join(args.output_dir, "labels.txt")
            with open(labels_path, "w") as f:
                for c in class_names:
                    f.write(c + "\n")
            print(f"Saved best model to {save_path} (ValAcc: {best_acc*100:.2f}%)")

    # plot curves
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="train loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot([a*100 for a in val_accuracies], label="val acc (%)")
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "training_plot.png")
    plt.savefig(plot_path)
    print("Training finished. Plot saved to:", plot_path)
    print("Best validation accuracy: {:.2f}%".format(best_acc*100))

if __name__ == "__main__":
    main()
