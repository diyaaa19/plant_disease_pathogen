## 🌿 Plant Disease Pathogen Classification

This project classifies plant diseases into pathogen categories using Deep Learning models.

### 📊 Dataset

* PlantVillage Dataset (Colored images)
* Mapped diseases to pathogen types:

  * Bacteria
  * Fungus
  * Oomycete
  * Virus
  * Nan

### 🧠 Models Used

* ResNet50 (Transfer Learning)
* EfficientNetB0 (Transfer Learning)

### 📈 Performance

| Model          | Accuracy |
| -------------- | -------- |
| ResNet50       | 54%      |
| EfficientNetB0 | 95%      |

### 📂 Project Structure

* Training scripts
* Prediction scripts
* Evaluation scripts
* Results

### ▶️ How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Train model:

```
python efficientnet_training.py
```

3. Predict:

```
python efficient_prediction.py
```



