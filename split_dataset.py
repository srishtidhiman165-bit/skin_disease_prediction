
import pandas as pd
import os
import shutil
from sklearn.model_selection import GroupShuffleSplit


# ---------------------------------------
# 1. File paths
# ---------------------------------------

metadata_path = r"C:\Users\dell\Downloads\HAM10000_metadata.csv"

source_folder = r"C:\Users\dell\Downloads\New folder\skin_disease_prediction\dataset"

output_folder = r"C:\Users\dell\Downloads\New folder\skin_disease_prediction\dataset_split"


# ---------------------------------------
# 2. Read metadata
# ---------------------------------------

df = pd.read_csv(metadata_path)

print("Metadata loaded!")
print("Total images:", len(df))


# ---------------------------------------
# 3. First split: 70% Train, 30% Temporary
# ---------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_index, temp_index = next(
    splitter.split(
        df,
        groups=df["lesion_id"]
    )
)

train_df = df.iloc[train_index].copy()
temp_df = df.iloc[temp_index].copy()


# ---------------------------------------
# 4. Second split:
#    Temporary → 15% Validation + 15% Test
# ---------------------------------------

splitter2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_index, test_index = next(
    splitter2.split(
        temp_df,
        groups=temp_df["lesion_id"]
    )
)

val_df = temp_df.iloc[val_index].copy()
test_df = temp_df.iloc[test_index].copy()


# ---------------------------------------
# 5. Create folders
# ---------------------------------------

splits = {
    "train": train_df,
    "validation": val_df,
    "test": test_df
}

disease_names = {
    "nv": "Melanocytic_nevi",
    "mel": "Melanoma",
    "bkl": "Benign_keratosis",
    "bcc": "Basal_cell_carcinoma",
    "akiec": "Actinic_keratoses",
    "vasc": "Vascular_lesions",
    "df": "Dermatofibroma"
}


for split_name in splits:

    for disease_name in disease_names.values():

        folder = os.path.join(
            output_folder,
            split_name,
            disease_name
        )

        os.makedirs(folder, exist_ok=True)


# ---------------------------------------
# 6. Copy images
# ---------------------------------------

for split_name, split_df in splits.items():

    print("\nProcessing:", split_name)

    for _, row in split_df.iterrows():

        image_id = row["image_id"]
        disease_code = row["dx"]

        disease_name = disease_names[disease_code]

        image_name = image_id + ".jpg"

        source = os.path.join(
            source_folder,
            disease_name,
            image_name
        )

        destination_folder = os.path.join(
            output_folder,
            split_name,
            disease_name
        )

        destination = os.path.join(
            destination_folder,
            image_name
        )

        if os.path.exists(source):

            shutil.copy2(source, destination)

        else:

            print("Image not found:", image_name)


# ---------------------------------------
# 7. Show final counts
# ---------------------------------------

print("\n================================")
print("DATASET SPLIT COMPLETE")
print("================================")

print("\nTrain images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))

print("\nTotal:", len(train_df) + len(val_df) + len(test_df))