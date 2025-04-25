import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
BASE_DATA_DIR = "img"
CSV_PATH = os.path.join(BASE_DATA_DIR, "data-min.csv")
LOW_RES_DIR = os.path.join(BASE_DATA_DIR, "low_res")
HIGH_RES_DIR = os.path.join(BASE_DATA_DIR, "high_res")
TARGET_SIZE = (128, 128)
INPUT_SHAPE = (*TARGET_SIZE, 3)
BATCH_SIZE = 4
EPOCHS = 50

# Data loading
try:
    data = pd.read_csv(CSV_PATH)
    data["low_res"] = data["low_res"].apply(lambda x: os.path.join(LOW_RES_DIR, str(x)))
    data["high_res"] = data["high_res"].apply(
        lambda x: os.path.join(HIGH_RES_DIR, str(x))
    )
    print(f"Loaded {len(data)} records.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Data Generator
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_low_gen = datagen.flow_from_dataframe(
    data,
    x_col="low_res",
    target_size=TARGET_SIZE,
    class_mode=None,
    batch_size=BATCH_SIZE,
    subset="training",
    seed=42,
    shuffle=True,
)

train_high_gen = datagen.flow_from_dataframe(
    data,
    x_col="high_res",
    target_size=TARGET_SIZE,
    class_mode=None,
    batch_size=BATCH_SIZE,
    subset="training",
    seed=42,
    shuffle=True,
)

val_low_gen = datagen.flow_from_dataframe(
    data,
    x_col="low_res",
    target_size=TARGET_SIZE,
    class_mode=None,
    batch_size=BATCH_SIZE,
    subset="validation",
    seed=42,
    shuffle=False,
)

val_high_gen = datagen.flow_from_dataframe(
    data,
    x_col="high_res",
    target_size=TARGET_SIZE,
    class_mode=None,
    batch_size=BATCH_SIZE,
    subset="validation",
    seed=42,
    shuffle=False,
)


# Helper to yield (input, target) pairs for model.fit
def paired_generator(low_gen, high_gen):
    while True:
        yield (next(low_gen), next(high_gen))


train_gen = paired_generator(train_low_gen, train_high_gen)
val_gen = paired_generator(val_low_gen, val_high_gen)

# Calculate steps
train_samples = train_low_gen.samples
val_samples = val_low_gen.samples
steps_per_epoch = train_samples // BATCH_SIZE
validation_steps = val_samples // BATCH_SIZE

if steps_per_epoch == 0 or validation_steps == 0:
    print("Error: Not enough samples for batch size. Adjust BATCH_SIZE or dataset.")
    exit()

# Autoencoder Model (No Skip Connections)
input_img = Input(shape=INPUT_SHAPE)
# Encoder
x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
x = MaxPooling2D((2, 2), padding="same")(x)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
encoded = MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = Conv2D(64, (3, 3), activation="relu", padding="same")(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(
    x
)  # Sigmoid for [0,1] output

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer="adam", loss="mean_squared_error")
autoencoder.summary()

# Training
print("\nStarting Training...")
history = autoencoder.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=validation_steps,
    verbose=1,
)
print("Training Finished.")

# Inference & Visualization
print("\nVisualizing one prediction...")
# Get one batch from the validation generator
val_low_batch, val_high_batch = next(val_gen)

# Predict
predicted_batch = autoencoder.predict(val_low_batch)

# Display the first image of the batch
idx_to_show = 0
if len(val_low_batch) > idx_to_show:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(val_low_batch[idx_to_show])
    axs[0].set_title("Low Res Input")
    axs[0].axis("off")

    axs[1].imshow(val_high_batch[idx_to_show])
    axs[1].set_title("High Res Ground Truth")
    axs[1].axis("off")

    axs[2].imshow(np.clip(predicted_batch[idx_to_show], 0.0, 1.0))  # Clip just in case
    axs[2].set_title("Predicted High Res")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()
else:
    print("Validation batch was empty, cannot visualize.")

print("\nScript finished.")
