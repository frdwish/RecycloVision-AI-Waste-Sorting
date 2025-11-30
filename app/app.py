# app/app.py
import sys, os
# make project root importable so "utils" can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from PIL import Image
from utils.utils import load_model, predict_from_file, get_device

st.set_page_config(page_title="Smart Waste Classifier", layout="centered")
st.title("♻️ Smart Waste Classifier (Upload Image)")

# locate model & labels (works when running from project root)
MODEL_PATH = os.path.join("model", "model.pth")
LABELS_PATH = os.path.join("model", "labels.txt")

if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    st.warning("Model or labels not found in model/ — please train first and ensure model/model.pth and model/labels.txt exist.")
else:
    device = get_device()
    model, labels, device = load_model(MODEL_PATH, LABELS_PATH, device=device)
    st.info(f"Model loaded. Classes: {', '.join(labels)}. Device: {device}")

uploaded_file = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # load image first (important)
    image = Image.open(uploaded_file).convert("RGB")

    # display image (use_container_width replaces deprecated use_column_width)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Predicting..."):
        label, conf = predict_from_file(uploaded_file, model, labels, device)
        conf_pct = int(conf * 100)
        st.success(f"**Category:** {label} — **Confidence:** {conf_pct}%")
        st.progress(conf_pct)

        bin_map = {
            'paper':'📘 Paper Bin — Blue',
            'plastic':'🟡 Plastic Bin — Yellow',
            'metal':'⚙️ Metal Bin — Grey',
            'glass':'🟢 Glass Bin — Green',
            'cardboard':'🟦 Cardboard — Blue',
            'trash':'🗑️ General Waste — Black'
        }
        key = label.lower()
        if key in bin_map:
            st.markdown(f"### Dispose in: {bin_map[key]}")
        else:
            st.info("No bin mapping for this category.")
