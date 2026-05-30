# Overview

This module implements a deep learning-based mosquito classification system to distinguish between Aedes and Non-Aedes mosquitoes.

It analyzes image data using trained models to support early identification of disease-carrying mosquitoes.

This is a classification module, not a medical or diagnostic system.

# What the Module Does

Classifies mosquito images into Aedes or Non-Aedes
Uses deep learning models (YOLO, CNN, EfficientNet)
Evaluates models based on accuracy, generalization, and bias
Provides stable predictions suitable for integration into larger systems

# Model Outcome

Some models achieved higher accuracy but suffered from overfitting or class bias
The final model (mosquito_model.keras) achieves ~80% accuracy with balanced and reliable performance
Model selection prioritized stability over raw accuracy

# Future Enhancements

Improve dataset balance and diversity
Apply data augmentation and class weighting
Use advanced validation techniques
Explore lightweight models for deployment