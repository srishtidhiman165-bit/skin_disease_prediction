
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import os

print("TensorFlow version:", tf.__version__)

# ==============================
# DATASET PATHS
# ==============================

train_path = r"C:\Users\dell\Downloads\New folder\skin_disease_prediction\dataset_balanced\train"

validation_path = r"C:\Users\dell\Downloads\New folder\skin_disease_prediction\dataset_split\validation"

# ==============================
# SETTINGS
# ==============================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 7
EPOCHS = 15

# ==============================
# LOAD DATASET
# ==============================

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names

print("\nClass names:")
print(class_names)

# ==============================
# PERFORMANCE
# ==============================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# ==============================
# EFFICIENTNETB0
# ==============================

print("\nLoading EfficientNetB0...")

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# ==============================
# BUILD MODEL
# ==============================

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

# ==============================
# COMPILE
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel ready!")
model.summary()

# ==============================
# CALLBACKS
# ==============================

os.makedirs("models", exist_ok=True)

checkpoint = ModelCheckpoint(
    "models/best_skin_disease_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001,
    verbose=1
)

# ==============================
# TRAIN MODEL
# ==============================

print("\n================================")
print("STARTING MODEL TRAINING")
print("================================")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)

# ==============================
# SAVE FINAL MODEL
# ==============================

model.save("models/skin_disease_model.keras")

print("\n================================")
print("TRAINING COMPLETE!")
print("================================")

print("\nBest model:")
print("models/best_skin_disease_model.keras")

print("\nFinal model:")
print("models/skin_disease_model.keras")