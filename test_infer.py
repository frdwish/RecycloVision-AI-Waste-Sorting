from utils.utils import load_model, predict_from_file
import os, sys

MODEL = "model/model.pth"
LABELS = "model/labels.txt"
IMG = sys.argv[1]  # pass image path as arg

model, labels, device = load_model(MODEL, LABELS)
label, conf = predict_from_file(IMG, model, labels, device)
print(f"Predicted: {label}  ({conf*100:.2f}%)")
