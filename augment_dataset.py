
import os
import random
import shutil
from PIL import Image, ImageEnhance, ImageOps


# ---------------------------------------
# 1. Paths
# ---------------------------------------

train_folder = r"C:\Users\dell\Downloads\New folder\skin_disease_prediction\dataset_split\train"

output_folder = r"C:\Users\dell\Downloads\New folder\skin_disease_prediction\dataset_balanced\train"


# ---------------------------------------
# 2. Minimum images required
# ---------------------------------------

TARGET_COUNT = 500


# ---------------------------------------
# 3. Create output folder
# ---------------------------------------

os.makedirs(output_folder, exist_ok=True)


# ---------------------------------------
# 4. Augmentation function
# ---------------------------------------

def augment_image(image):

    # Random horizontal flip
    if random.random() < 0.5:
        image = ImageOps.mirror(image)

    # Random vertical flip
    if random.random() < 0.2:
        image = ImageOps.flip(image)

    # Random rotation
    angle = random.uniform(-20, 20)
    image = image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC
    )

    # Random brightness
    brightness = random.uniform(0.85, 1.15)
    image = ImageEnhance.Brightness(image).enhance(brightness)

    # Random contrast
    contrast = random.uniform(0.85, 1.15)
    image = ImageEnhance.Contrast(image).enhance(contrast)

    return image


# ---------------------------------------
# 5. Process every disease
# ---------------------------------------

diseases = [
    folder for folder in os.listdir(train_folder)
    if os.path.isdir(os.path.join(train_folder, folder))
]


for disease in diseases:

    source_disease_folder = os.path.join(
        train_folder,
        disease
    )

    output_disease_folder = os.path.join(
        output_folder,
        disease
    )

    os.makedirs(
        output_disease_folder,
        exist_ok=True
    )

    # Get original images
    images = [
        file for file in os.listdir(source_disease_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    original_count = len(images)

    print("\n--------------------------------")
    print("Disease:", disease)
    print("Original images:", original_count)

    # ---------------------------------------
    # Copy original images
    # ---------------------------------------

    for image_name in images:

        source = os.path.join(
            source_disease_folder,
            image_name
        )

        destination = os.path.join(
            output_disease_folder,
            image_name
        )

        shutil.copy2(source, destination)

    # ---------------------------------------
    # If already 500+, no augmentation
    # ---------------------------------------

    if original_count >= TARGET_COUNT:

        print("Already 500+ images.")
        print("No augmentation required.")

        continue

    # ---------------------------------------
    # Generate augmented images
    # ---------------------------------------

    required = TARGET_COUNT - original_count

    print("Augmented images required:", required)

    for i in range(required):

        # Select random original image
        image_name = random.choice(images)

        source = os.path.join(
            source_disease_folder,
            image_name
        )

        try:

            image = Image.open(source).convert("RGB")

            # Apply augmentation
            augmented = augment_image(image)

            # New filename
            new_name = (
                f"aug_{disease}_{i + 1:04d}.jpg"
            )

            destination = os.path.join(
                output_disease_folder,
                new_name
            )

            augmented.save(
                destination,
                "JPEG",
                quality=95
            )

        except Exception as e:

            print(
                "Error processing:",
                image_name,
                e
            )


# ---------------------------------------
# 6. Final count
# ---------------------------------------

print("\n================================")
print("DATASET BALANCING COMPLETE")
print("================================")

for disease in diseases:

    folder = os.path.join(
        output_folder,
        disease
    )

    count = len([
        file for file in os.listdir(folder)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    print(f"{disease}: {count} images")

print("\nBalanced dataset location:")
print(output_folder)