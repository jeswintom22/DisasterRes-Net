import os
import io
import base64
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template

import torch
import torch.nn.functional as F
import timm
import joblib
from torchvision import transforms as T

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "saved_models"
IMG_SIZE = (299, 299)
OBJECTIVES = ("disaster_type", "damage")

MODELS = {}
LABEL_ENCODERS = {}

def get_transforms():
    return T.Compose([
        T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        score = output[:, class_idx].squeeze()
        score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.size(2), input_tensor.size(3)))
            
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()[0]
        
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=0).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val
            
        return heatmap

def init_models():
    print("[INFO] Loading PyTorch Checkpoints...")
    for obj in OBJECTIVES:
        le_path = os.path.join(MODEL_DIR, f"le_{obj}.joblib")
        model_path = os.path.join(MODEL_DIR, f"model_{obj}.pth")
        
        if not os.path.exists(le_path) or not os.path.exists(model_path):
            print(f"[WARN] Missing model checkpoints for {obj}")
            continue
            
        le = joblib.load(le_path)
        LABEL_ENCODERS[obj] = le
        
        model = timm.create_model("inception_resnet_v2", pretrained=False, num_classes=len(le.classes_))
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        MODELS[obj] = model
    print("[INFO] Model Loading Complete!")

init_models()

def generate_heatmap_overlay(original_img_cv, heatmap, alpha=0.5):
    heatmap_resized = cv2.resize(heatmap, (original_img_cv.shape[1], original_img_cv.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, original_img_cv, 1 - alpha, 0)
    return overlay

def get_target_layer(model):
    try:
        return dict([*model.named_modules()])['conv2d_7b']
    except KeyError:
        for name, mod in reversed(list(model.named_modules())):
            if isinstance(mod, torch.nn.Conv2d):
                return mod
    return list(model.modules())[-1]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files["file"]
    try:
        img_bytes = file.read()
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image format: {e}"}), 400
    
    original_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    transform = get_transforms()
    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    results = {}
    
    for obj in OBJECTIVES:
        if obj not in MODELS:
            results[obj] = {"error": "Model not trained"}
            continue
            
        model = MODELS[obj]
        le = LABEL_ENCODERS[obj]
        
        with torch.no_grad():
            with torch.amp.autocast(device.type):
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)
        
        pred_label = le.inverse_transform([pred_idx.item()])[0]
        confidence = float(conf.item()) * 100
        
        target_layer = get_target_layer(model)
        cam = GradCAM(model, target_layer)
        
        with torch.enable_grad():
            input_tensor.requires_grad = True
            heatmap = cam(input_tensor, class_idx=pred_idx.item())
        
        overlay_bgr = generate_heatmap_overlay(original_cv, heatmap)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        overlay_pil = Image.fromarray(overlay_rgb)
        
        buffered = io.BytesIO()
        overlay_pil.save(buffered, format="JPEG", quality=85)
        overlay_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        results[obj] = {
            "prediction": pred_label,
            "confidence": f"{confidence:.2f}",
            "heatmap": f"data:image/jpeg;base64,{overlay_b64}"
        }
        
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
