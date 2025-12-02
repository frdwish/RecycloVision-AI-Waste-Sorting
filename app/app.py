#importing
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from PIL import Image
from utils.utils import load_model, predict_from_file, get_device


# Page Config
st.set_page_config(
    page_title="RecycloVision – Smart AI Waste Sorting",
    layout="centered",
    page_icon="♻️",
)

# Modern Header
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='font-size:40px;'>♻️ RecycloVision – AI Waste Sorting</h1>
        <p style='font-size:18px; color:#cccccc;'>
            Upload a waste image and instantly get the correct disposal bin.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# Loading Model
MODEL_PATH = os.path.join("model", "model.pth")
LABELS_PATH = os.path.join("model", "labels.txt")

if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    st.error("⚠️ Model not found. Please train the model first.")
    st.stop()

device = get_device()
model, labels, device = load_model(MODEL_PATH, LABELS_PATH, device=device)


# Upload Section
st.markdown("### 📤 Step 1: Upload an image")
uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    help="Upload a picture of waste to classify",
)

if uploaded_file is not None:


# Image Preview Card
    image = Image.open(uploaded_file).convert("RGB")
    st.markdown("### 🖼️ Preview")
    st.image(image, use_container_width=True)

# Prediction
    with st.spinner("Analyzing image..."):
        label, conf = predict_from_file(uploaded_file, model, labels, device)
        conf_pct = int(conf * 100)

# Clean Prediction Card
    st.markdown(
        f"""
        <div style="
            background-color:#1a1a1a;
            padding:18px;
            border-radius:12px;
            margin-top:20px;
            border:1px solid #333;
        ">
            <h3 style='color:#4CAF50;'>✓ Prediction Successful</h3>
            <p style='font-size:18px;'><b>Category:</b> {label.capitalize()}</p>
            <p style='font-size:18px;'><b>Confidence:</b> {conf_pct}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(conf_pct)

# BIN Recommendation (Bold + Colorful)
    bin_map = {
        'paper':      ('Paper Bin — Blue', "#1e90ff"),
        'plastic':    ('Plastic Bin — Yellow', "#ffd500"),
        'metal':      ('Metal Bin — Grey', "#8c8c8c"),
        'glass':      ('Glass Bin — Green', "#1ebc6f"),
        'cardboard':  ('Cardboard Bin — Blue', "#1e90ff"),
        'trash':      ('General Waste — Black', "#2b2b2b"),
    }

    key = label.lower()

    if key in bin_map:
        bin_name, color = bin_map[key]

        st.markdown(
            f"""
            <div style="
                margin-top:25px;
                padding:20px;
                background-color:{color};
                border-radius:12px;
                text-align:center;
                color:black;
                font-size:22px;
                font-weight:bold;
            ">
                🚮 Dispose in: {bin_name}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("No bin mapping available for this category.")


# Footer
st.markdown(
    """
    <hr>
    <div style='text-align:center; opacity:0.6; font-size:13px;'>
    Built with ❤️ using PyTorch + Streamlit<br>
    Helping improve recycling through AI
    </div>
    """,
    unsafe_allow_html=True
)
