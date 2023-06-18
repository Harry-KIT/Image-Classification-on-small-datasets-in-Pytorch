import torch
import torchvision.transforms as transforms
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import shufflenet_v2_x2_0

# Define the device to run the model on (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transforms to apply to the input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the test dataset
test_dataset = ImageFolder(root='/path/to/test/data', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model
model = shufflenet_v2_x2_0()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Move the model to the device
model.to(device)

# Load the pre-trained weights
model.load_state_dict(torch.load('results/weights.pth', map_location=device))

# Set the model to evaluation mode
model.eval()

# Initialize the variables to store the total and correct predictions
total = 0
correct = 0

# Iterate over the test data
with torch.no_grad():
    for images, labels in test_loader:
        # Move the images and labels to the device
        images = images.to(device)
        labels = labels.to(device)

        # Get the model's predictions
        output = model(images)

        # Get the predictions with the highest confidence
        _, predictions = torch.max(output, dim=1)

        # Update the total and correct predictions
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

# Calculate the accuracy of the model
accuracy = correct / total

print('Accuracy of the model on the test data: {:.2f}%'.format(accuracy * 100))

