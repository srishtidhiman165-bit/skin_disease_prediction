
import tensorflow as tf
import numpy as np

# ==============================
# MODEL
# ==============================

model_path = r"models\best_skin_disease_model.keras"

# ==============================
# IMAGE
# ==============================

image_path = r"C:\Users\dell\Desktop\ISIC_0024307.jpg"

# ==============================
# CLASS NAMES
# ==============================

class_names = [
    "Actinic_keratoses",
    "Basal_cell_carcinoma",
    "Benign_keratosis",
    "Dermatofibroma",
    "Melanocytic_nevi",
    "Melanoma",
    "Vascular_lesions"
]

# ==============================
# LOAD MODEL
# ==============================

print("Loading model...")

model = tf.keras.models.load_model(model_path)

print("Model loaded successfully!")

# ==============================
# LOAD IMAGE
# ==============================

image = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224)
)

image_array = tf.keras.utils.img_to_array(image)
image_array = tf.expand_dims(image_array, 0)

# ==============================
# PREDICTION
# ==============================

print("\nMaking prediction...")

predictions = model.predict(image_array, verbose=0)

predicted_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_index]
confidence = predictions[0][predicted_index] * 100

# ==============================
# RESULT
# ==============================

print("\n================================")
print("SKIN DISEASE PREDICTION")
print("================================")

print("Predicted Disease:", predicted_class)
print(f"Confidence: {confidence:.2f}%")

print("================================")