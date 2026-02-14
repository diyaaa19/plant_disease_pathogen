import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ------------------ PATHS ------------------
MODEL_PATH = "colored_pathogen_classifier_efficientnet.h5"   # change if needed
IMAGE_DIR = "PlantVillage/color"                          # change if needed
EXCEL_PATH = "plant_disease_filled.xlsx"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ------------------ Load Model ------------------
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully.\n")

# ------------------ Load Excel ------------------
df = pd.read_excel(EXCEL_PATH, header=0)
df.columns = (
    df.columns.astype(str)
    .str.replace('\u00a0', ' ', regex=False)
    .str.replace('\n', '', regex=False)
    .str.strip()
    .str.lower()
)

disease_col = "disease name"
pathogen_col = "pathogen responsible"

df[pathogen_col] = df[pathogen_col].astype(str).str.strip().str.title()
df[disease_col] = df[disease_col].astype(str).str.strip()

# ------------------ Build Test Dataset ------------------
image_paths = []
labels = []

for disease, pathogen in zip(df[disease_col], df[pathogen_col]):
    folder_name = disease.replace(" ", "_").replace("/", "_")
    folder_path = os.path.join(IMAGE_DIR, folder_name)

    if os.path.exists(folder_path):
        for img in os.listdir(folder_path):
            if img.lower().endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(folder_path, img))
                labels.append(pathogen)

data = pd.DataFrame({"image": image_paths, "label": labels})

print(f"Total test images: {len(data)}\n")

# ------------------ Image Generator ------------------
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_gen = test_datagen.flow_from_dataframe(
    data,
    x_col="image",
    y_col="label",
    target_size=IMG_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ------------------ Predictions ------------------
print("\nGenerating predictions...")
pred_probs = model.predict(test_gen, verbose=1)

y_pred = np.argmax(pred_probs, axis=1)
y_true = test_gen.classes
class_names = list(test_gen.class_indices.keys())

# ------------------ Confusion Matrix ------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - EfficientNet")
plt.tight_layout()
plt.savefig("efficientnet_confusion_matrix.png")
plt.show()

print("\nConfusion Matrix saved as efficientnet_confusion_matrix.png")

# ------------------ Classification Report ------------------
report = classification_report(y_true, y_pred, target_names=class_names)

print("\nClassification Report:\n")
print(report)

with open("efficientnet_classification_report.txt", "w") as f:
    f.write(report)

print("Classification report saved as efficientnet_classification_report.txt")

# ------------------ Overall Metrics ------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")

print("\nOverall Metrics:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

with open("efficientnet_test_metrics.txt", "w") as f:
    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1 Score  : {f1:.4f}\n")

print("Metrics saved as efficientnet_test_metrics.txt")

print("\n✅ Evaluation Complete.")
