# SolderWorks: AI-Powered Solder Defect Detection

SolderWorks is a simple machine learning project that detects and classifies the quality of solder joints on circuit boards. It was trained using TensorFlow and later integrated into Python for real-world testing.

The project demonstrates how AI can automate visual inspection tasks — a small step toward smarter electronics manufacturing.

---

## Overview

The model classifies images of solder joints into four categories:

- ✅ **Good**
- ⚠️ **Cold**
- ❌ **Insufficient**
- and detects solder bridges !

It uses a **TensorFlow/Keras** model trained on a dataset of labeled solder joint photos. The model was exported and optimized for inference in Python.

---

## 🧠 How It Works

1. A photo of a solder joint is captured or uploaded.
2. The image is resized, normalized, and fed into the trained neural network.
3. The model outputs a prediction label and a confidence score.

This setup can be expanded to include live image capture from a camera or integrated with inspection hardware for automated quality control.

---

## Skills & Tools

- TensorFlow / Keras  
- Python (for model integration)  
- Image preprocessing with Pillow  
- Data collection & labeling  
- Basic ML pipeline optimization  

