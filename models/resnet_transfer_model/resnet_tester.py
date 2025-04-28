import time
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "Separated_Dataset"
test_dir = os.path.join(data_dir, "test") 
saved_dir = "saved_models" 
output_file = "all_variants_results.txt" 

# prepare test data
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]), 
])
test_ds = datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

variants = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

# load models 
resnets = {} 
for name in variants:
    model = getattr(models, name)(weights=None)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(test_ds.classes))
    model = model.to(device) 

    path = os.path.join(saved_dir, name, "resnet_model.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(path + " not found")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    resnets[name] = model
    print(name + " loaded")

print("\nAll variants loaded\n")

# collect true labels once 
print("collecting true labels...") 
all_labels = [] 
with torch.no_grad(): 
    for _, labels in test_loader:
        all_labels.extend(labels.numpy())

print("collected " + str(len(all_labels)) + " labels\n") 


all_preds = {}
times = {}

with torch.no_grad(): 
    for name, model in resnets.items():
        print("running inference for " + name + "...")
        start = time.time()
        preds = []
        for images, _ in test_loader: 
            images = images.to(device)  
            out = model(images)
            _, p = torch.max(out, 1)	
            preds.extend(p.cpu().numpy())
        elapsed = time.time() - start
        times[name] = elapsed 
        all_preds[name] = preds
        print("finished " + name + ", got " + str(len(preds)) +
              " preds in " + str(round(elapsed,2)) + "s")
 
print("\nrun complete for all variants\n")


# write results
with open(output_file, "w") as f:
    for name in variants:
        preds = all_preds[name]
        labels = all_labels
        total = len(labels)

        # accuracy
        correct = 0
        for i in range(total):
            if preds[i] == labels[i]:
                correct += 1
        acc = correct * 100.0 / total

        f.write("=== " + name + " ===\n")

        f.write("Inference time: " + str(round(times[name],2)) + "s\n")
        f.write("Accuracy: " + str(round(acc,2)) + "%\n\n")


        # actual counts
        classes = test_ds.classes
        actual_counts = {}

        for c in classes:
            actual_counts[c] = 0

        for lab in labels:
            cls = classes[lab]
            actual_counts[cls] += 1
        f.write("Actual class counts:\n")

        for c in classes:
            pct = actual_counts[c] * 100.0 / total
            f.write("  " + c + ": " + str(actual_counts[c]) + 
                    " (" + str(round(pct,2)) + "%)\n") 

        # predicted counts
        pred_counts = {}
        
	for c in classes:
            pred_counts[c] = 0
        
	for p in preds:
            cls = classes[p]
            pred_counts[cls] += 1
        f.write("\nPredicted class counts:\n")
        
	for c in classes:
            pct = pred_counts[c] * 100.0 / total
            f.write("  " + c + ": " + str(pred_counts[c]) +
                    " (" + str(round(pct,2)) + "%)\n")

        # file lists
        correct_files = []
        wrong_files = []
        for i in range(total):
            path, lab = test_ds.samples[i]
            if preds[i] == lab:
                correct_files.append(path)
            else:
                wrong_files.append(path)

        f.write("\nCorrectly classified (" + str(len(correct_files)) +
                "/" + str(total) + "):\n ")
        
	for cf in correct_files:
            f.write("  " + cf + "\n")

        f.write("\nMisclassified (" + str(len(wrong_files)) +
                "/" + str(total) + "):\n")

        for wf in wrong_files:
            f.write("  " + wf + "\n ")
        f.write("\n")

print("All results written to " + output_file)