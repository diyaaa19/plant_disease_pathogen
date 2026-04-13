import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ======================
# 1. LOAD DATA (FIXED)
# ======================
EXCEL_PATH = "plant_disease_filled.xlsx"
IMAGE_DIR = "PlantVillage/color"

df = pd.read_excel(EXCEL_PATH)

df.columns = (
    df.columns.astype(str)
    .str.replace('\u00a0', ' ', regex=False)
    .str.replace('\n', '', regex=False)
    .str.strip()
    .str.lower()
)

disease_col = "disease name"
pathogen_col = "pathogen responsible"

# Clean labels
df[pathogen_col] = df[pathogen_col].astype(str).str.strip().str.title()
df[disease_col] = df[disease_col].astype(str).str.strip()

# Remove null rows (IMPORTANT)
df = df[df[pathogen_col].notna()]

# ======================
# 2. BUILD DATASET
# ======================
image_paths, labels = [], []

for disease, pathogen in zip(df[disease_col], df[pathogen_col]):
    folder_name = disease.replace(" ", "_").replace("/", "_")
    folder_path = os.path.join(IMAGE_DIR, folder_name)

    if os.path.exists(folder_path):
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(folder_path, img_file))
                labels.append(pathogen)

data = pd.DataFrame({"image": image_paths, "label": labels})

# ======================
# 3. BINARY LABEL
# ======================
data["binary_label"] = data["label"].apply(
    lambda x: "Healthy" if x == "Nan" else "Diseased"
)

# ======================
# 4. SPLIT
# ======================
train_df, test_df = train_test_split(data, test_size=0.1, stratify=data["binary_label"], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["binary_label"], random_state=42)

# ======================
# 5. GENERATORS
# ======================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.1,
    horizontal_flip=True
).flow_from_dataframe(
    train_df, x_col="image", y_col="binary_label",
    target_size=IMG_SIZE, class_mode="categorical", batch_size=BATCH_SIZE
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    val_df, x_col="image", y_col="binary_label",
    target_size=IMG_SIZE, class_mode="categorical", batch_size=BATCH_SIZE
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    test_df, x_col="image", y_col="binary_label",
    target_size=IMG_SIZE, class_mode="categorical", batch_size=BATCH_SIZE, shuffle=False
)

# ======================
# 6. CLASS WEIGHTS
# ======================
labels_array = train_df["binary_label"]
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_array),
    y=labels_array
)
class_weight_dict = dict(enumerate(class_weights))

# ======================
# 7. MODEL (FINE-TUNE)
# ======================
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224,224,3))

for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers[-30:]:
    layer.trainable = True

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(2, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ======================
# 8. TRAIN + AUTO SAVE
# ======================
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint("best_binary_model.h5", save_best_only=True, monitor="val_accuracy")
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# ======================
# 9. FINAL SAVE
# ======================
model.save("final_binary_model.h5")

# ======================
# 10. EVALUATION
# ======================
y_pred = model.predict(test_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_gen.classes

report = classification_report(y_true, y_pred_classes)

with open("classification_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# ======================
# 11. FEATURE EXTRACTION
# ======================
feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
features = feature_model.predict(test_gen)

np.save("features.npy", features)
np.save("labels.npy", y_true)

# ======================
# 12. PCA
# ======================
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)

plt.figure()
plt.scatter(pca_result[:,0], pca_result[:,1], c=y_true)
plt.title("PCA")
plt.savefig("pca.png")
plt.close()

# ======================
# 13. TSNE
# ======================
tsne = TSNE(n_components=2, perplexity=30)
tsne_result = tsne.fit_transform(features)

plt.figure()
plt.scatter(tsne_result[:,0], tsne_result[:,1], c=y_true)
plt.title("TSNE")
plt.savefig("tsne.png")
plt.close()

print("✅ EVERYTHING SAVED SUCCESSFULLY")
