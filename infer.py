import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt 
import os

def predict_pil(image: Image.Image):
    # Load model architecture and weights
    num_classes = len(class_names)
    model = CNNModel(num_classes=num_classes)
    model.load_state_dict(torch.load("tomato_disease_model.pth", map_location="cpu"))
    model.eval()

    # Preprocessing (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = class_names[predicted_idx.item()]
        confidence = confidence.item()
    return {"label": predicted_class, "confidence": confidence}

# Define the CNN Model (Must match the trained model)
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
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class names (Ensure this matches your dataset)
class_names = [
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

num_classes = len(class_names)  # Set the correct number of classes

# Load the trained model
model = CNNModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("tomato_disease_model.pth", map_location=device))
model.eval()  # Set model to evaluation mode

# Define image transformations (Must match the training process)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to preprocess the image
def preprocess_image(image_path):
    # Make image path robust to working directory
    if not os.path.isabs(image_path):
        # Get project root (parent of this file)
        project_root = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(project_root, image_path)
    image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB mode
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Function to make a prediction
def predict(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    predicted_class = class_names[predicted.item()]
    print(f"Predicted Class: {predicted_class}")
    if predicted_class == "Tomato___Bacterial_spot":
        print("Treat seed with New Improved Ceresan or use disease-free plants. ")
    return predicted_class

def visualize_feature_maps(model, image_path):
    image = preprocess_image(image_path)  # robust path
    outputs = []

    def hook(module, input, output):
        outputs.append(output.detach().cpu())

    # Register hook on each convolutional layer
    model.conv1.register_forward_hook(hook)
    model.conv2.register_forward_hook(hook)
    model.conv3.register_forward_hook(hook)

    # Forward pass
    model(image)

    # Plot feature maps from each conv layer
    # for idx, feature_map in enumerate(outputs):
    #     fig, axes = plt.subplots(1, 6, figsize=(15, 5))
    #     fig.suptitle(f"Feature Maps from Conv{idx+1}", fontsize=14)
    #     for i in range(6):  # show first 6 feature maps
    #         axes[i].imshow(feature_map[0, i].numpy(), cmap="gray")
    #         axes[i].axis("off")
    #     plt.show()

# Example usage
visualize_feature_maps(model, "data/img1.jpg")

# Run inference
if __name__ == "__main__":
    image_path = os.path.join("data", "img1.jpg")
    predict(image_path)