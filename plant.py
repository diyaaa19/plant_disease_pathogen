import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

EXCEL_PATH = "plant_disease_filled.xlsx"
IMAGE_DIR = "PlantVillage/color"   
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

df = pd.read_excel(EXCEL_PATH, header=0)
df.columns = (
    df.columns
    .astype(str)
    .str.replace('\u00a0', ' ', regex=False)  # remove non-breaking spaces
    .str.replace('\n', '', regex=False)       # remove newlines
    .str.strip()                              # remove leading/trailing spaces
    .str.lower()                              # standardize case
)

print(df.columns.tolist())

disease_col = "disease name"
pathogen_col = "pathogen responsible"

df[pathogen_col] = df[pathogen_col].astype(str).str.strip().str.title()
df[disease_col] = df[disease_col].astype(str).str.strip()

image_paths = []
labels = []

for disease, pathogen in zip(df[disease_col], df[pathogen_col]):
    folder_name = disease.replace(" ", "_").replace("/", "_")
    folder_path = os.path.join(IMAGE_DIR, folder_name)
    if os.path.exists(folder_path):
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(folder_path, img_file))
                labels.append(pathogen)

data = pd.DataFrame({"image": image_paths, "label": labels})
print(f" Total images found: {len(data)}")
print(f" Pathogen classes: {data['label'].unique()}")

train_df, test_df = train_test_split(data, test_size=0.1, stratify=data["label"], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["label"], random_state=42)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df, x_col="image", y_col="label",
    target_size=IMG_SIZE, class_mode="categorical",
    batch_size=BATCH_SIZE, shuffle=True
)

val_gen = val_datagen.flow_from_dataframe(
    val_df, x_col="image", y_col="label",
    target_size=IMG_SIZE, class_mode="categorical",
    batch_size=BATCH_SIZE, shuffle=False
)

test_gen = test_datagen.flow_from_dataframe(
    test_df, x_col="image", y_col="label",
    target_size=IMG_SIZE, class_mode="categorical",
    batch_size=BATCH_SIZE, shuffle=False
)

base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False  # freezing base layers

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(len(train_gen.class_indices), activation="softmax")(x)


model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
    ModelCheckpoint("colored_pathogen_resnet.h5", save_best_only=True, monitor="val_accuracy")
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(test_gen)
print(f" Test Accuracy: {test_acc:.4f}")

model.save("colored_pathogen_resnet_classifier.h5")
print("Model saved as colored_pathogen_resnet_classifier.h5")
