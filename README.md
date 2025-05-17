Bone Fracture Detection from Shoulder X-rays
============================================

Project Overview:
-----------------
This project focuses on automated bone fracture detection from shoulder X-ray images using deep learning. The goal is to assist radiologists by providing a second opinion tool that can classify images as 'fractured' or 'non-fractured'.

We experimented with several models:
- EfficientNet (pretrained on ImageNet)
- ResNet50 (pretrained)
- DenseNet
- A custom CNN architecture
- Ensemble learning for robust performance

Dataset:
--------
- The dataset contains shoulder X-ray images labeled as fractured or non-fractured.
- The data was preprocessed by resizing, normalizing using ImageNet statistics, and augmenting to balance the classes.
- Images were split into training, validation, and test sets.

Directory Structure:
--------------------
CV_Project.ipynb            → Jupyter notebook with full training and evaluation code  
/shoulder_xray/             → Folder containing images and label folders (e.g., fractured, normal)  
/models/                    → Trained model weights (optional for saving)  
/output/                    → Folder to save evaluation results, confusion matrices, etc.

Model Features:
---------------
1. EfficientNet, ResNet, DenseNet:
   - Transfer learning with fine-tuning
   - Added custom classifier heads
   - Normalized with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]

2. Custom CNN:
   - Built from scratch
   - Includes Conv → ReLU → MaxPool → FC → Softmax layers

3. Ensemble Learning:
   - Combines predictions from all models
   - Uses majority voting or averaging for final prediction
   - Improves robustness across varying X-ray image quality

Evaluation Metrics:
-------------------
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

All results are printed in the notebook and saved (optional) for further analysis.

Dependencies:
-------------
- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-learn
- PIL

To install dependencies:
pip install torch torchvision numpy matplotlib scikit-learn pillow

Usage:
------
1. Run CV_Project.ipynb from start to finish in a Jupyter environment.
2. Update dataset paths if needed.
3. You can visualize predictions, performance metrics, and model comparisons inside the notebook.

Future Work:
------------
- Integrate explainable AI (e.g., Grad-CAM) to visualize fracture regions
- Deploy model with a simple Flask or Streamlit interface
- Expand dataset with more diverse fracture types and patient demographics

Author:
-------
Goutham Darapogu  
darapogugoutham@gmail.com  
Master’s Student, Computer Science  
Texas A&M University – Corpus Christi
