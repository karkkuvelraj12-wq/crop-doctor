"""
Crop Doctor: train MobileNetV3-Small on PlantVillage and export a quantized TFLite model.

Run in Google Colab with a GPU runtime (Runtime > Change runtime type > T4 GPU).
Outputs: crop_disease.tflite and labels.txt (copy both into the app's assets/ folder).
"""
# !pip install -q tensorflow-datasets   (uncomment in Colab)
import os
import tensorflow as tf
import tensorflow_datasets as tfds

print("GPU:", tf.config.list_physical_devices("GPU"))

IMG, BATCH = 224, 32

# 45% of the data for training, 10% validation, 10% test (faster; raise to 80% for best accuracy)
(train, val, test), info = tfds.load(
    "plant_village",
    split=["train[:45%]", "train[80%:90%]", "train[90%:]"],
    as_supervised=True,
    with_info=True,
)
names = info.features["label"].names
N = len(names)
with open("labels.txt", "w") as f:
    f.write("\n".join(names))


def prep(x, y):
    # MobileNetV3 has built-in preprocessing, so keep pixel values in 0-255.
    return tf.image.resize(x, (IMG, IMG)), y


A = tf.data.AUTOTUNE
train_ds = train.map(prep, A).shuffle(1000).batch(BATCH).prefetch(A)
val_ds = val.map(prep, A).batch(BATCH).prefetch(A)
test_ds = test.map(prep, A).batch(BATCH).prefetch(A)

base = tf.keras.applications.MobileNetV3Small(
    input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet"
)
base.trainable = False

infer = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(N, activation="softmax"),
])

# Augmentation helps bridge the gap between lab photos and real field photos.
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomBrightness(0.3),
    tf.keras.layers.RandomContrast(0.3),
])
inp = tf.keras.Input((IMG, IMG, 3))
model = tf.keras.Model(inp, infer(augment(inp)))


def compile_(lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


# Phase 1: train the new classification head
compile_(1e-3)
model.fit(train_ds, validation_data=val_ds, epochs=2)

# Phase 2: fine-tune the top layers of the base model
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
compile_(1e-4)
model.fit(train_ds, validation_data=val_ds, epochs=3)

print("Test accuracy:", model.evaluate(test_ds)[1])


# Export to quantized TFLite (small and fast on low-end phones)
def representative_data():
    for x, _ in train.map(prep).batch(1).take(200):
        yield [x]


converter = tf.lite.TFLiteConverter.from_keras_model(infer)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data
tflite_model = converter.convert()

with open("crop_disease.tflite", "wb") as f:
    f.write(tflite_model)
print(f"Saved crop_disease.tflite ({len(tflite_model) / 1e6:.1f} MB)")

# In Colab, download the outputs:
# from google.colab import files
# files.download("crop_disease.tflite"); files.download("labels.txt")
