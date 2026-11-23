# Facial-Emotion-Recognition
# Facial Image Emotion Classification Using Deep Learning and Attention Mechanisms

## Overview
This project focuses on classifying facial images into emotion categories using deep learning architectures. It explores a custom CNN, attention-enhanced pretrained models, and a Vision Transformer (ViT), followed by deployment through a Flask web application.

## Project Workflow
1. **Dataset Analysis:** Examine dataset dimensions, class distribution, image properties, and pixel intensity statistics.
2. **Exploratory Data Analysis:** Create eight visualizations to analyse class distribution, image dimensions, aspect ratios, pixel intensities, and sample images.
3. **Preprocessing:** Apply gamma correction, Non-Local Means denoising, image resizing, normalization, and training-time data augmentation.
4. **Dataset Splitting:** Use stratified splitting with 70% training, 15% validation, and 15% testing.
5. **Model Development:** Train and evaluate four deep learning architectures:
   - Custom CNN
   - MobileNetV3-MHA
   - VGG17-SA
   - Vision Transformer (ViT)
6. **Model Evaluation:** Evaluate using Accuracy, Precision, Recall, F1-Score, and Brier Score.
7. **Visualization and Comparison:** Generate learning curves, classification reports, confusion matrices, ROC curves, and model comparison plots.
8. **Deployment:** Develop a Flask web application for facial emotion classification.

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy and Pandas
- Matplotlib and Seaborn
- Flask

## Project Structure
```text
Facial-Image-Emotion-Classification/
│
├── Code.ipynb
├── Results/
│   ├── EDA/
│   ├── Model_Evaluation/
│   └── Model_Comparison/
│
└── Web/
    ├── app.py
    ├── requirements.txt
    ├── model/
    ├── templates/
    └── static/
```

## Objective
To analyse facial image characteristics, develop and compare multiple deep learning architectures, and deploy an image-based emotion classification system using Flask.
