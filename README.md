# Cattle-Vision

An end-to-end deep learning system that detects cattle in images, draws a bounding box around it and classifies its breed from 15 cattle breeds with confidence score.

## Streamlit App Link
[Cattle Vision](https://cattlebreedclassifier-u2marrrgzv6fyzfnu9tahh.streamlit.app/)

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
