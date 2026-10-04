
import pandas as pd
import os
import shutil

# -----------------------------
# 1. Paths
# -----------------------------

metadata_path = r"C:\Users\dell\Downloads\HAM10000_metadata.csv"

image_folder_1 = r"C:\Users\dell\Downloads\HAM10000_images_part_1"
image_folder_2 = r"C:\Users\dell\Downloads\HAM10000_images_part_2"

output_folder = r"C:\Users\dell\Downloads\New folder\skin_disease_prediction\dataset"


# -----------------------------
# 2. Read metadata
# -----------------------------

df = pd.read_csv(metadata_path)

print("Metadata loaded successfully!")
print("Total images:", len(df))


# -----------------------------
# 3. Disease names
# -----------------------------

disease_names = {
    "nv": "Melanocytic_nevi",
    "mel": "Melanoma",
    "bkl": "Benign_keratosis",
    "bcc": "Basal_cell_carcinoma",
    "akiec": "Actinic_keratoses",
    "vasc": "Vascular_lesions",
    "df": "Dermatofibroma"
}


# -----------------------------
# 4. Create disease folders
# -----------------------------

for disease_code, disease_name in disease_names.items():

    folder = os.path.join(output_folder, disease_name)

    os.makedirs(folder, exist_ok=True)


# -----------------------------
# 5. Copy images
# -----------------------------

copied = 0
not_found = 0

for index, row in df.iterrows():

    image_id = row["image_id"]
    disease_code = row["dx"]

    disease_name = disease_names[disease_code]

    image_name = image_id + ".jpg"

    source = os.path.join(image_folder_1, image_name)

    if not os.path.exists(source):
        source = os.path.join(image_folder_2, image_name)

    destination_folder = os.path.join(
        output_folder,
        disease_name
    )

    destination = os.path.join(
        destination_folder,
        image_name
    )

    if os.path.exists(source):

        shutil.copy2(source, destination)

        copied += 1

    else:

        print("Image not found:", image_name)

        not_found += 1


# -----------------------------
# 6. Final result
# -----------------------------

print("\n-----------------------------")
print("Dataset organization complete!")
print("-----------------------------")

print("Images copied:", copied)
print("Images not found:", not_found)

print("\nDataset location:")
print(output_folder)
print("\nFinal disease-wise image count:")

for disease_code, disease_name in disease_names.items():

    folder = os.path.join(output_folder, disease_name)

    image_count = len([
        file for file in os.listdir(folder)
        if file.lower().endswith(".jpg")
    ])

    print(f"{disease_name}: {image_count}")