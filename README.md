# ♻️ RecycloVision – AI Waste Sorting System
Deep Learning–powered smart waste classification using **ResNet-18 + Streamlit**  
Automatically identifies waste type and recommends the **correct disposal bin**.

---

## Overview
**RecycloVision** is an AI-based waste classification system that recognizes **six types of waste** and instantly tells the user which disposal bin to use.  
It is built with **PyTorch, Transfer Learning, and Streamlit**, optimized for **Apple Silicon (M1/M2)**.

This project supports cleaner recycling, reduces contamination, and makes waste segregation easier for everyone — even non-technical users.

---

## Problem Statement
Incorrect waste disposal leads to:
- Contamination of recyclable materials  
- Increased landfill waste  
- Higher recycling costs  
- Slower waste processing  

Manual segregation is slow and inconsistent.  
Thus, we need an **automated, accurate, and easy-to-use waste classifier.**

---

## Objective
Build a **real-time, high-accuracy waste classifier** that:
- Recognizes six waste categories  
- Runs fast using ResNet-18 transfer learning  
- Provides correct **bin color suggestions**  
- Works through a clean, intuitive Streamlit UI  

---

## Features
- **Real-time waste classification**
- **6 waste categories:** cardboard, glass, metal, paper, plastic, trash
- **Disposal Bin Recommendation**
- 📘 Blue – Paper / Cardboard  
- 🟡 Yellow – Plastic  
- 🟢 Green – Glass  
- ⚙️ Grey – Metal  
- 🗑️ Black – General Waste  
- **93.66% validation accuracy**
- **Optimized for M1/M2 GPUs (MPS backend)**
- **User-friendly Streamlit interface**
- **Complete training → evaluation → inference pipeline**

---

## Project Structure
```
RecycloVision-AI-Waste-Sorting/
│
├── app/
│ └── app.py # Streamlit Web App
│
├── utils/
│ └── utils.py # Preprocessing, model loader, prediction
│
├── training/
│ └── train.py # ResNet-18 training script
│
├── model/
│ ├── labels.txt # Class labels
│ └── training_plot.png # Accuracy/Loss graph
│
├── test_images/ # Sample images
│
├── dataset/ # Empty placeholder (dataset ignored)
│ └── .gitkeep
│
├── test_infer.py # CLI inference script
├── requirements.txt # Dependencies
├── LICENSE # MIT License
└── README.md # Documentation
```


---

## Installation

### 1️. Clone the Repository
```
git clone https://github.com/frdwish/RecycloVision-AI-Waste-Sorting.git
cd RecycloVision-AI-Waste-Sorting
```
### 2️. Create a Virtual Environment
```
python3 -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows
```
### 3️. Install Requirements
```
pip install -r requirements.txt
```
### 3.Run the Web App
```
streamlit run app/app.py
```
### 4. Run Inference (Test Single Image)
```
python test_infer.py path/to/image.jpg
🧠 Model Training (Optional)
```
## Model Training(optional)

```
cd training
python train.py \
  --data_dir ../dataset/dataset-resized/ \
  --output_dir ../model/ \
  --epochs 20 \
  --batch_size 32
```
## Results

| Metric                 | Score                          |
|------------------------|--------------------------------|
| Validation Accuracy    | **93.66%**                     |
| Model                  | ResNet-18 (Transfer Learning)  |
| Device                 | Apple M2 GPU (MPS Backend)     |


## Future Enhancements

- Deploy to HuggingFace Spaces
- Add webcam live detection
- Expand dataset with more classes
- ONNX/TFLite model export for mobile apps
- Add multilingual UI

## License

- This project is licensed under the MIT License.
- You are free to use, modify, and distribute it for personal or commercial use.

## Acknowledgements

- TrashNet Dataset
- PyTorch team
- Streamlit community
- Apple M2/MPS backend

## Support

- If you find this project helpful, consider giving it a ⭐ on GitHub!

## App Screenshots



