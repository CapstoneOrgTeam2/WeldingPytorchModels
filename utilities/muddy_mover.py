import os
import csv
import shutil

os.makedirs("../models/resnet_transfer_model/muddy_files", exist_ok=True)

with open("../models/resnet_transfer_model/database/bad_with_good_label_images.csv", "r", newline="") as f:
    reader = csv.DictReader(f)
 
    for row in reader:
        src = row["path"]

        if src != "NOT FOUND" and os.path.exists(src): 
            shutil.move(src, os.path.join("../models/resnet_transfer_model/muddy_files", row["file_name"]))
