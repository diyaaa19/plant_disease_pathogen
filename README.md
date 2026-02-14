## 🌿 Plant Disease Pathogen Classification

This project classifies plant diseases into pathogen categories using Deep Learning models.

## 🎥 Demo Video

[![Watch the demo](https://img.youtube.com/vi/gLk8t4FylC0/0.jpg)](https://www.youtube.com/watch?v=gLk8t4FylC0)


### 📊 Dataset

* PlantVillage Dataset (Colored images)
* Source: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
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



