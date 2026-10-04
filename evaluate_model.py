
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ==============================
# PATHS
# ==============================

model_path = r"models/best_skin_disease_model.keras"

test_path = r"C:\Users\dell\Downloads\New folder\skin_disease_prediction\dataset_split\test"

# ==============================
# SETTINGS
# ==============================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# ==============================
# LOAD MODEL
# ==============================

print("Loading trained model...")

model = tf.keras.models.load_model(model_path)

print("Model loaded successfully!")

# ==============================
# LOAD TEST DATASET
# ==============================

print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names

print("\nClass names:")
print(class_names)

# ==============================
# PREDICTIONS
# ==============================

print("\nMaking predictions...")

y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ==============================
# ACCURACY
# ==============================

accuracy = np.mean(y_true == y_pred)

print("\n================================")
print("TEST ACCURACY")
print("================================")

print(f"Test Accuracy: {accuracy * 100:.2f}%")

# ==============================
# CLASSIFICATION REPORT
# ==============================

print("\n================================")
print("CLASSIFICATION REPORT")
print("================================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)

# ==============================
# CONFUSION MATRIX
# ==============================

print("\n================================")
print("CONFUSION MATRIX")
print("================================")

cm = confusion_matrix(y_true, y_pred)

print(cm)

print("\n================================")
print("EVALUATION COMPLETE!")
print("================================")