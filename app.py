# imports
import streamlit as st
import torch
import torch.nn as nn
from  torchvision import models, transforms
from ultralytics import YOLO
import cv2
import json
import numpy as np
from PIL import Image
import io

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load classifier
@st.cache_resource
def load_classifier():
    num_classes = 15
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_classifier()

# load labels
with open("breed_labels.json") as f:
    breed_labels = json.load(f)



# load yolo
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")
yolo = load_yolo()

# transform
IMG_SIZE = 224
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# classifier function
def classifier(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    trans_image = val_tf(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(trans_image)
        probs = torch.softmax(logits, dim = 1)
        conf, idx = probs.max(dim = 1)
    
    return breed_labels[idx.item()], conf.item()

# full pipeline
def pipeline(image):
    output_image = image.copy()
    predictions = []
    detections = yolo.predict(image, classes = 19, conf=0.4)

    for box in detections[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if x2 <= x1 or y2 <= y1:
            continue

        crop_image = image[y1:y2, x1:x2]
        if crop_image.size == 0:
            continue

        breed, confidence = classifier(crop_image)
        predictions.append((breed, confidence))

        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0,255,0), 2)

        text_y = max(25, y1 - 10)
        label = f"{breed} ({confidence:.2f})"

        cv2.putText(output_image, label, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    return output_image, predictions

# UI
st.sidebar.divider()
st.sidebar.markdown("### Supported Cattle Breeds")
st.sidebar.markdown("\n".join([f"- {str.capitalize(breed)}" for breed in breed_labels]))
st.set_page_config(page_title="Cattle Breed Detector", layout="wide")

st.title("Cattle-Vision")
st.write("Upload an image to detect cattle and identify their breeds. To upload another image click on **Browse files**. You can view the supported cattle breeds in the sidebar.")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Input Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    with st.spinner("Detecting cattle and predicting breeds..."):
        result_image, predictions = pipeline(image)
    
    
    with col2:
        st.markdown("### Detection Result")
        st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        if len(predictions) == 0:
            st.warning("No cattle detected.")
        else:
            for i, (breed, confidence) in enumerate(predictions, 1):
                st.write(f"**Cattle {i}:** {breed} — Confidence: {confidence*100:.1f}%")
    
        _, buffer = cv2.imencode(".jpg", result_image)
        st.download_button(
            "Download Result",
            buffer.tobytes(),
            "cattle_result.jpg",
            "image/jpeg"
        )

    
