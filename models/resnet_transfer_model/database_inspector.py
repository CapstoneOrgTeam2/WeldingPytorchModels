import os
import csv
import sqlite3

DATABASE_FILE = "dataset_database_1.db"

def main():
    # Quick script to check stuff about the dataset
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    # images total
    cursor.execute("SELECT COUNT(*) FROM images")
    total_images = cursor.fetchone()[0]

    # classification breakdown
    cursor.execute("""
        SELECT current_classification, COUNT(*) 
        FROM images 
        GROUP BY current_classification
    """)
    classification_counts = cursor.fetchall()


    # sum up bounding boxes
    cursor.execute("""
        SELECT 
            SUM(bad_boxes), 
            SUM(good_boxes), 
            SUM(defect_boxes), 
            SUM(total_boxes)
        FROM images
    """)
    bad_sum, good_sum, defect_sum, total_sum = cursor.fetchone()


    # muddy query - find all images classified as bad but with good bounding boxes too
    cursor.execute("""
        SELECT COUNT(*)
        FROM images
        WHERE current_classification='Bad'
          AND good_boxes>0
    """)

    bad_with_good = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM images
        WHERE current_classification='Good'
          AND (bad_boxes>0 OR defect_boxes>0)
    """)
    good_with_bad = cursor.fetchone()[0]


    # dump images that are physically in Bad_Weld but have good boxes
    cursor.execute("""
        SELECT file_name
        FROM images
        WHERE label_text='Bad_Weld'
          AND good_boxes>0
    """)
    muddy_filenames = [row[0] for row in cursor.fetchall()]

    folder_candidates = [
        os.path.join("Separated_Dataset", "train", "Bad_Weld"),
        os.path.join("Separated_Dataset", "valid", "Bad_Weld")
    ]

    with open("bad_with_good_label_images.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "possible_path"])
        for fn in muddy_filenames:
            found = "NOT FOUND"
            for fold in folder_candidates:
                path = os.path.join(fold, fn)
                if os.path.exists(path):
                    found = path
                    break
            writer.writerow([fn, found])
    conn.close()

    print(f"\nTotal images: {total_images}")
    print("--- Current_classification ---")
    for cl, ct in classification_counts:
        print(f"{cl}: {ct}")

    print("\n--- Box Sums ---")
    print(f"Bad: {bad_sum if bad_sum else 0}")
    print(f"Good: {good_sum if good_sum else 0}")
    print(f"Defect: {defect_sum if defect_sum else 0}")
    print(f"Total: {total_sum if total_sum else 0}")


    print("\n\n--- Muddied data ---")
    print(f"Bad but has Good boxes: {bad_with_good}") # Images that are in bad_weld folder but have good_weld boxes
    print(f"Good but has Bad/Defect boxes: {good_with_bad}") # This should never be above 0.

    print("\nExported 'bad_with_good_label_images.csv' for muddy data (images in Bad_Weld with Good boxes.")

if __name__ == "__main__":
    main()
