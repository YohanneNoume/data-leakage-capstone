from flask import Flask, request, jsonify
import torch
import torchvision
import base64
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import os

app = Flask(__name__)

device = torch.device("cpu")
model = None

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "resnet18_leakfree.pth"

# Preprocessing (defined once)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

def load_model():
    global model
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully from {MODEL_PATH}")

load_model()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]
    img = Image.open(BytesIO(base64.b64decode(data))).convert("L")

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_class = torch.argmax(probs).item()

    return jsonify({
        "diagnosis": "Pneumonia" if pred_class == 1 else "Normal",
        "confidence": float(probs[pred_class])
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9696))
    app.run(host="0.0.0.0", port=port, debug=False)