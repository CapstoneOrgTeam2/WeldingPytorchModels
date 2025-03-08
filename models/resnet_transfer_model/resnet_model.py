import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def main():
    data_dir = r"C:\Users\gutte\OneDrive\Documents\School_2025\Capstone\MyFirstTryProject\DataProcessingProject\Separated_Dataset"
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    # Define classes and training parameters
    num_classes = 2
    batch_size = 8
    num_epochs = 10
    learning_rate = 0.001

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((640, 640)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # No augmentation for validation/testing
    test_transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
 
    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = torchvision.datasets.ImageFolder(root=valid_dir, transform=test_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataset  = torchvision.datasets.ImageFolder(root=test_dir,  transform=test_transform)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False) 
    
    # Bear a cuda!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ResNet
    import torchvision.models as models
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all layers except final (fully connected) layer
    for param in resnet.parameters():
        param.requires_grad = False

    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, num_classes)

    # Move model to device
    resnet = resnet.to(device)

    # Set up loss function and optimizer (remember: only training the final layer)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.fc.parameters(), lr=learning_rate)

    # training loop
    for epoch in range(num_epochs):
        resnet.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = resnet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        resnet.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = resnet(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Acc: {accuracy:.2f}%")

    # Testing
    resnet.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = resnet(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # COmpute accuracy
    total_test = len(all_labels)
    correct_test = sum(p == a for p, a in zip(all_preds, all_labels))
    
    test_accuracy = 100.0 * correct_test / total_test
    print(f"\nTest Accuracy: {test_accuracy:.2f}%")

    class_names = test_dataset.classes  # e.g. ["Bad_Weld", "Good_Weld"]
    predicted_labels = [class_names[p] for p in all_preds]
    actual_labels    = [class_names[a] for a in all_labels]

    # actual vs predicted classes
    from collections import Counter
    actual_counts = Counter(actual_labels)
    predicted_counts = Counter(predicted_labels)

    print("\nCount of Actual Classes:")
    for c in class_names:
        print(f"  {c}: {actual_counts[c]}")

    print("\nCount of Predicted Classes:")
    for c in class_names:
        print(f"  {c}: {predicted_counts[c]}")

    correct_files = []
    wrong_files = []
    
    with open("test_results_resnet.csv", "w") as f:
        f.write("ImagePath,Actual,Predicted\n")
        for i, (path, label_idx) in enumerate(test_dataset.samples):
            actual_class = class_names[label_idx]
            predicted_class = class_names[all_preds[i]]
            f.write(f"{path},{actual_class},{predicted_class}\n")

            if all_preds[i] == label_idx:
                correct_files.append(path)
            else:
                wrong_files.append(path)

    print("\nTest results saved to 'test_results_resnet.csv'")
    print("\nCorrectly classified images:")
    
    for cf in correct_files:
        print(" ", cf)

    print("\nMisclassified images:")
    
    for wf in wrong_files:
        print(" ", wf)

    # Save model
    torch.save(resnet.state_dict(), "resnet_model.pth")
    print("Model saved as 'resnet_model.pth'")

if __name__ == "__main__":
    main()
