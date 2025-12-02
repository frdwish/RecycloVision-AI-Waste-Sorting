# utils/utils.py
import torch
from torchvision import transforms
from PIL import Image
import os



# Return MPS (Apple GPU) if available, otherwise CPU
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# Load trained model + labels from disk
def load_model(model_path='model/model.pth', labels_path='model/labels.txt', device=None):
    if device is None:
        device = get_device()
    # build ResNet-18 architecture and load weights
    from torchvision.models import resnet18
    # read labels to determine num classes
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    with open(labels_path, 'r') as f:
        labels = [l.strip() for l in f.readlines() if l.strip()]
    num_classes = len(labels)
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, labels, device

# Preprocess PIL image into model-ready tensor
def preprocess_pil(img_pil, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return transform(img_pil).unsqueeze(0)

# Predict class + confidence from image file or stream
def predict_from_file(image_path_or_fileobj, model, labels, device, img_size=224):
    # image_path_or_fileobj can be a file path or file-like object (Streamlit upload)
    if hasattr(image_path_or_fileobj, "read"):
        img = Image.open(image_path_or_fileobj).convert("RGB")
    else:
        img = Image.open(image_path_or_fileobj).convert("RGB")
    # Preprocess and move to device
    input_tensor = preprocess_pil(img, img_size=img_size).to(device)
    with torch.no_grad():
        out = model(input_tensor)
        probs = torch.nn.functional.softmax(out, dim=1)[0]
        idx = probs.argmax().item()
        return labels[idx], float(probs[idx])
