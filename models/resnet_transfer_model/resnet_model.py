import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def main():
    data_dir = r"C:\Users\gutte\OneDrive\Documents\School_2025\Capstone\WeldingPytorchModels\models\resnet_transfer_model\Separated_Dataset"
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    # classes and training parameters
    num_classes = 2
    batch_size = 8
    num_epochs = 10
    learning_rate = 0.001

    # data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((640, 640)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # none for validation/testing
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
    test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Bear a cuda!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import torchvision.models as models
    # define new model variants
    model_variants = {
        "resnet101": models.ResNet101_Weights.DEFAULT,
        "resnet152": models.ResNet152_Weights.DEFAULT
    }

    for model_name, weights in model_variants.items():
        print("\nTraining with", model_name)
        resnet = getattr(models, model_name)(weights=weights)

        # freeze all layers except final (fully connected) layer
        for param in resnet.parameters():
            param.requires_grad = False

        in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features, num_classes)
        resnet = resnet.to(device)

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
            print(f"{model_name} Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Acc: {accuracy:.2f}%")

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

        total_test = len(all_labels)
        correct_test = sum(p == a for p, a in zip(all_preds, all_labels))
        test_accuracy = 100.0 * correct_test / total_test
        print(f"\n{model_name} Test Accuracy: {test_accuracy:.2f}%")

        class_names = test_dataset.classes  # e.g. ["Bad_Weld", "Good_Weld"]
        predicted_labels = [class_names[p] for p in all_preds]
        actual_labels = [class_names[a] for a in all_labels]

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

        results_csv = f"test_results_{model_name}.csv"
        with open(results_csv, "w") as f:
            f.write("ImagePath,Actual,Predicted\n")
            for i, (path, label_idx) in enumerate(test_dataset.samples):
                actual_class = class_names[label_idx]
                predicted_class = class_names[all_preds[i]]
                f.write(f"{path},{actual_class},{predicted_class}\n")
                if all_preds[i] == label_idx:
                    correct_files.append(path)
                else:
                    wrong_files.append(path)

        print(f"\nTest results saved to '{results_csv}'")
        print("\nCorrectly classified images:")
        for cf in correct_files:
            print(" ", cf)
        print("\nMisclassified images:")
        for wf in wrong_files:
            print(" ", wf)

        # Save model in a new folder for this variant
        save_folder = os.path.join("saved_models", model_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        model_path = os.path.join(save_folder, "resnet_model.pth")
        torch.save(resnet.state_dict(), model_path)
        print(f"Model saved as '{model_path}'")


if __name__ == "__main__":
    main()
