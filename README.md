# Facial Emotion Recognition Using Deep Learning

## Overview
This project develops a facial emotion recognition system using deep learning to classify facial images into five emotion categories: Angry, Fear, Happy, Sad, and Suprise. It includes dataset analysis, image preprocessing, model training, performance evaluation, and Flask-based deployment.

## Dataset
- **Dataset:** [Human Face Emotions](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions)
- **Total Images:** 59,099
- **Emotion Classes:** 5

## Project Workflow
1. **Basic Statistics and EDA:** Analyse dataset dimensions, class distribution, image properties, and pixel intensity.
2. **Preprocessing:** Resize images, convert to grayscale and three channels, apply image sharpening, and normalize pixel values.
3. **Dataset Splitting:** Stratified split into 70% training, 15% validation, and 15% testing.
4. **Model Development:** Implement and evaluate:
   - Custom CNN
   - MobileNetV3 with Multi-Head Attention (MHA)
   - ResNet18
5. **Model Evaluation:** Measure Accuracy, Precision, Recall, F1-Score, and Brier Score, supported by evaluation visualizations.
6. **Deployment:** Integrate the trained ResNet18 model into a Flask web application for facial emotion prediction.

## Model Performance

| Model | Accuracy | F1-Score |
|---|---:|---:|
| CNN | 53.94% | 49.21% |
| MobileNetV3-MHA | 84.50% | 84.46% |
| ResNet18 | 89.92% | 89.99% |

## Technologies Used
Python, PyTorch, Torchvision, OpenCV, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Flask, and Pillow.

## Project Structure
```text
Facial Emotion Recognition/
│
├── Code.ipynb
├── Results/
│   ├── EDA and model result images
│   └── Web application screenshots
│
└── WebApp/
    ├── app.py
    ├── best_emotion_model.pth
    ├── requirements.txt
    ├── templates/
    │   └── index.html
    └── static/
        └── uploads/
```

## Objective
To develop a deep learning-based facial emotion recognition system and deploy the trained model through an interactive Flask web application.
