import os
import sqlite3

DATABASE_FILE = "dataset_database_1.db"

# 0 -> Bad_Weld, 1 -> Good_Weld
DATASET_PATHS = {
    "Separated_Dataset/train/Bad_Weld": 0,
    "Separated_Dataset/train/Good_Weld": 1,
    "Separated_Dataset/valid/Bad_Weld": 0,
    "Separated_Dataset/valid/Good_Weld": 1,
}

conn = sqlite3.connect(DATABASE_FILE)
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS images")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT UNIQUE,
        label_text TEXT,
        current_classification TEXT,
        total_boxes INTEGER,
        bad_boxes INTEGER,
        good_boxes INTEGER,
        defect_boxes INTEGER
    ) """
)

for folder, classification_value in DATASET_PATHS.items():

    # By default, 0 => "Bad", 1 => "Good"
    default_classification = "Bad" if classification_value == 0 else "Good"
    label_text = "Bad_Weld" if classification_value == 0 else "Good_Weld"

    if not os.path.exists(folder):
        print(f"Folder does not exist: {folder}")
        continue

    for image_name in os.listdir(folder):
        if image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            # Start with folder-based classification
            current_classification = default_classification

            # Build path to .txt label in "labels/" folder
            base_name = os.path.splitext(image_name)[0]
            label_file_name = base_name + ".txt"
            label_file_path = os.path.join("labels", label_file_name)

            bad_count = 0
            good_count = 0
            defect_count = 0
            total_count = 0

            if os.path.exists(label_file_path):
                with open(label_file_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue  # skip malformed lines
                        class_id = int(parts[0])
                        if class_id == 0:
                            bad_count += 1
                        elif class_id == 1:
                            good_count += 1
                        elif class_id == 2:
                            defect_count += 1
                        total_count += 1
            else:
                print(f"Label file not found for {image_name} at {label_file_path}")

            # If there's ANY defect box, override classification to 'Bad'
            if defect_count > 0:
                current_classification = "Bad"


            try:
                cursor.execute(
                    """
                    INSERT INTO images (
                        file_name,
                        label_text,
                        current_classification,
                        total_boxes,
                        bad_boxes,
                        good_boxes,
                        defect_boxes
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?) 
                    """,
                    (
                        image_name,
                        label_text,
                        current_classification,
                        total_count,
                        bad_count,
                        good_count,
                        defect_count
                    ),
                )
            except sqlite3.IntegrityError:
                print(f"Skipping entry for {image_name}")

conn.commit()
conn.close()

print("Database created/updated successfully!")
