# Cattle Breed Detection & Classification

An end-to-end deep learning system that detects cattle in images and classifies their breed.

## Live App
[Streamlit Deployment Link Here]

## Features
- YOLOv8 object detection for cattle localization
- EfficientNet classifier for breed prediction
- Real-time web interface built with Streamlit
- Supports multiple image formats
- Downloadable prediction results

## Model Architecture
Detection: YOLOv8n  
Classification: EfficientNet-B0 (Transfer Learning)

## Workflow
Image → Detection → Cropping → Breed Classification → Result Display

## Tech Stack
Python, PyTorch, OpenCV, Streamlit, Ultralytics YOLO
