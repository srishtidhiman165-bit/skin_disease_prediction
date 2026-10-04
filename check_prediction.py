
import pandas as pd

# ==============================
# IMAGE NAME
# ==============================

image_name = "ISIC_0024307"

# ==============================
# METADATA FILE
# ==============================

metadata_path = r"C:\Users\dell\Downloads\HAM10000_metadata.csv"

# ==============================
# DISEASE MAPPING
# ==============================

disease_mapping = {
    "nv": "Melanocytic_nevi",
    "mel": "Melanoma",
    "bkl": "Benign_keratosis",
    "bcc": "Basal_cell_carcinoma",
    "akiec": "Actinic_keratoses",
    "vasc": "Vascular_lesions",
    "df": "Dermatofibroma"
}

# ==============================
# LOAD METADATA
# ==============================

df = pd.read_csv(metadata_path)

# Find image
result = df[df["image_id"] == image_name]

# ==============================
# DISPLAY RESULT
# ==============================

if len(result) == 0:
    print("Image not found in metadata.")
else:
    actual_code = result.iloc[0]["dx"]
    actual_disease = disease_mapping[actual_code]

    print("\n================================")
    print("ACTUAL DISEASE")
    print("================================")

    print("Image:", image_name)
    print("Actual Disease:", actual_disease)
    print("Disease Code:", actual_code)

    print("================================")