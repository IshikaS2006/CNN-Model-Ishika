import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# SAME transforms as in train.py but without augmentation for testing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- CNN Model (copied exactly from train.py) ---
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Path to test dataset
test_path = "data"  # Make sure your 5 images are in subfolders like in training
test_dataset = ImageFolder(root=test_path, transform=transform)
num_classes = len(test_dataset.classes)
print(f"Test classes: {test_dataset.classes}")

# Load trained model
model = CNNModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("tomato_disease_model.pth", map_location=device))
model.eval()

# Data loader for test set
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Accuracy calculation
test_correct, test_total = 0, 0
predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

        predictions.append((test_dataset.classes[predicted.item()], 
                            test_dataset.classes[labels.item()]))

# Final accuracy
test_accuracy = test_correct / test_total
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Show predictions
print("\nPredictions:")
for idx, (pred, actual) in enumerate(predictions):
    print(f"Image {idx+1}: Predicted = {pred} | Actual = {actual}")
