import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import torchvision.models as models
from collections import Counter


def main():
 
    data_dir = r"C:\Users\gutte\OneDrive\Documents\School_2025\Capstone\MyFirstTryProject\DataProcessingProject\Separated_Dataset"
    test_dir = os.path.join(data_dir, "test")
    

    test_transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2

    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    for param in resnet.parameters():
        param.requires_grad = False
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, num_classes)
    resnet = resnet.to(device)

    model_path = "resnet_model.pth"
    if not os.path.exists(model_path):
        result = f"Model file '{model_path}' not found.\n"
        with open("resnet_tester_results.txt", "w") as f:
            f.write(result)
        print(result)
        return
    result = f"Loading model from '{model_path}'\n"
    resnet.load_state_dict(torch.load(model_path))
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
    correct_test = sum(p == a for p, 
        a in zip(all_preds,
        all_labels))
        
    test_accuracy = 100.0 * correct_test / total_test
    result += f"\nTest Accuracy: {test_accuracy:.2f}%\n"

    class_names = test_dataset.classes
    actual_labels = [class_names[a] for a in all_labels]
    predicted_labels = [class_names[p] for p in all_preds]


    actual_counts = Counter(actual_labels)
    predicted_counts = Counter(predicted_labels)

    result += "\nCount of Actual Classes:\n"
    
    for c in class_names:
        pct = 100.0 * actual_counts[c] / total_test
        result += f"  {c}: {actual_counts[c]} ({pct:.2f}%)\n"
   
    result += "\nCount of Predicted Classes:\n"
  
    for c in class_names:
        pct = 100.0 * predicted_counts[c] / total_test
        result += f"  {c}: {predicted_counts[c]} ({pct:.2f}%)\n"

    correct_files = []
    wrong_files = []
    for i, (path, label_idx) in enumerate(test_dataset.samples):
        if all_preds[i] == label_idx:
            correct_files.append(path)
        else:
            wrong_files.append(path)
 
    correct_pct = 100.0 * len(correct_files) / total_test
    wrong_pct = 100.0 * len(wrong_files) / total_test

    result += f"\nCorrectly classified images: {len(correct_files)} ({correct_pct:.2f}%)\n"
   
   for cf in correct_files:
        result += f"  {cf}\n"
    result += f"\nMisclassified images: {len(wrong_files)} ({wrong_pct:.2f}%)\n"
   
   for wf in wrong_files:
        result += f"  {wf}\n"

    with open("resnet_tester_results.txt", "w") as f:
        f.write(result) 
    print(result)

if __name__ == "__main__":
    main()