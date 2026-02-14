import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("colored_pathogen_resnet_classifier.h5")
print(" Model loaded successfully!")

class_labels = [
    "Bacteria",
    "Fungus",
    "Nan",
    "Oomycete",
    "Virus"
]

def predict_disease(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class_index]
    confidence = np.max(prediction[0]) * 100

    
    print(f"\n Image: {img_path}")
    print(f" Predicted Class: {predicted_label}")
    print(f" Confidence: {confidence:.2f}%")

    img_disp = image.load_img(img_path)
    plt.imshow(img_disp)
    plt.axis("off")
    plt.title(f"Predicted: {predicted_label} ({confidence:.2f}%)", fontsize=14, color='green')
    plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide main tkinter window

    print("\n Please select an image for prediction:")
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        print(" No file selected.")
    elif not os.path.exists(file_path):
        print(f" File not found: {file_path}")
    else:
        predict_disease(file_path)
