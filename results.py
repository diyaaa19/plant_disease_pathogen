import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

EXCEL_PATH = "plant_disease_filled.xlsx"
IMAGE_DIR = "PlantVillage/color"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

df = pd.read_excel(EXCEL_PATH)
df.columns = df.columns.str.strip().str.lower()
disease_col = "disease name"
pathogen_col = "pathogen responsible"
df[disease_col] = df[disease_col].astype(str).str.strip()
df[pathogen_col] = df[pathogen_col].astype(str).str.strip().str.title()

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
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    data,
    test_size=0.1,
    stratify=data["label"],
    random_state=42
)


model = load_model("colored_pathogen_resnet.h5")

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_dataframe(
    test_df, x_col="image", y_col="label",
    target_size=IMG_SIZE, class_mode="categorical",
    batch_size=BATCH_SIZE, shuffle=False
)


test_loss, test_acc = model.evaluate(test_gen)
print(f"Test Accuracy: {test_acc:.4f}")

os.makedirs("results", exist_ok=True)


y_true = test_gen.classes
y_pred = model.predict(test_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
class_labels = list(test_gen.class_indices.keys())

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix")
plt.savefig("results/colored_pathogen_resnet.png")
plt.close()

# Classification Report
report = classification_report(y_true, y_pred_classes, target_names=class_labels)
with open("results/colored_pathogen_resnet_classification_report.txt", "w") as f:
    f.write(report)

# Test Metrics
with open("results/colored_pathogen_resnet_test_metrics.txt", "w") as f:
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")


print("All results saved in 'results/' folder.")
