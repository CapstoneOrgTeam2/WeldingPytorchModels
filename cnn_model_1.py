import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)   # First conv layer
        self.pool = nn.MaxPool2d(2, 2)                    # Pooling layer
        self.conv2 = nn.Conv2d(6, 16, 5)  # Second conv layer

        self.flatten_size = 16 * 157 * 157  # Adjusted for 640x640

        self.fc1 = nn.Linear(self.flatten_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)  # Output layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)     # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer
        return x

def main():
    # Step 1: Load dataset
    data_dir = "this_drive:/this_project/Separated_Dataset" # UpdateMe :>
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")
    num_classes = 3

    batch_size = 8
    num_epochs = 5
    learning_rate = 0.001

    # Transformations: resize, convert to tensor, normalize
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create dataset from folder, create dataloader
    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    valid_dataset = torchvision.datasets.ImageFolder(root=valid_dir, transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Step 2: Create Model
    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(num_classes=3).to(device)

    # Step 3: Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Step 4: Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Step 5: Validate Model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Acc: {accuracy:.2f}%")

    # Step 5 cont. testing
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    '''
    I modified this tutorial:
    Tutorial link: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    The numbered sentences in quotes are directly pulled from the tutorial

    1. "Load and normalize the training and validation datasets using torchvision"
        - "The output of torchvision datasets are PILImage images of range [0, 1].
            We transform them to Tensors of normalized range [-1, 1]."

    2. "Define a Convolutional Neural Network"
        - Modified to take 640x640 images

    3. "Define loss function and optimizer"
        - Uses a "Classification Cross-Entropy" loss and SGD with momentum

    4. Train the network on the training data
        - We simply have to loop over our data iterator, and feed the inputs to the network and optimize.
    
    5. Evaluate the network on the validation and test data
        - We will check this by predicting the class label that the neural network outputs, and checking it against the ground-truth.
            If the prediction is correct, we add the sample to the list of correct predictions.
    '''
    main()
