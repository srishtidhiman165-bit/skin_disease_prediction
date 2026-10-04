
import os

train_folder = r"C:\Users\dell\Downloads\New folder\skin_disease_prediction\dataset_balanced\train"

print("================================")
print("BALANCED DATASET CHECK")
print("================================")

total = 0

for disease in sorted(os.listdir(train_folder)):
    disease_folder = os.path.join(train_folder, disease)

    if not os.path.isdir(disease_folder):
        continue

    images = [
        file for file in os.listdir(disease_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    count = len(images)
    total += count

    print(f"{disease}: {count} images")

print("--------------------------------")
print("Total training images:", total)
print("================================")