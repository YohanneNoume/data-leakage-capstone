import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import sys
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = models.resnet18(pretrained=False)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model.to(device)

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py image.png")
        sys.exit(1)

    image_path = sys.argv[1]
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = BASE_DIR / "artifacts" / "resnet18_leakfree.pth"
    model = load_model(MODEL_PATH)

    image = preprocess_image(image_path).to(device)

    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = "Pneumonia" if pred == 1 else "Normal"
    print(f"Prediction: {label} ({confidence:.3f})")