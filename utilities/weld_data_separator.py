import os
import shutil

# ID mappings (from data.yaml)
class_mapping = {
    0: "Bad_Weld",
    1: "Good_Weld",
    2: "Defect"
}

dataset_path = "Unmodified_Weld_defect_dataset_v2"
output_path = "New_Separated_Dataset"

# make the folders
for split in ["train", "valid", "test"]:
    for class_name in class_mapping.values():
        os.makedirs(os.path.join(output_path, split, class_name), exist_ok=True)

def classify_and_move(split):
    images_path = os.path.join(dataset_path, split, "images")
    labels_path = os.path.join(dataset_path, split, "labels")

    total_images = len(os.listdir(images_path))
    class_counts = {class_name: 0 for class_name in class_mapping.values()}

    print(f"\nStarting {split} dataset ({total_images} images)...")

    for i, image_file in enumerate(os.listdir(images_path), start=1):
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(labels_path, label_file)

        # we will use good as default state
        category = "Good_Weld"

        # read label
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path, "r") as file:
                lines = file.readlines()

            # get IDs in label file
            class_ids = {int(line.split()[0]) for line in lines}

            # sort
            if 2 in class_ids:
                category = "Defect"
            elif 0 in class_ids:
                category = "Bad_Weld"
            elif 1 in class_ids:
                category = "Good_Weld"

        # move image
        src = os.path.join(images_path, image_file)
        dest = os.path.join(output_path, split, category, image_file)
        shutil.copy(src, dest)


        # Track counts
        class_counts[category] += 1

        # Print progress (satisfying)
        if i % 50 == 0 or i == total_images:
            print(f"  Processed {i}/{total_images} images...")
    print(f"  -> {class_counts['Bad_Weld']} Bad Welds, {class_counts['Good_Weld']} Good Welds, {class_counts['Defect']} Defects classified.")

# Do the splits
for split in ["train", "valid", "test"]:
    classify_and_move(split)
print("\nJob's done")