from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from src.model import build_model, get_weights
from src.utils import load_class_names


st.set_page_config(page_title="Poketmon Classifier", layout="centered")

st.title("Poketmon Classifier")

checkpoint_path = st.sidebar.text_input(
    "Checkpoint",
    value="outputs/resnet18_head_only/best_model.pt",
)
backbone = st.sidebar.selectbox(
    "Backbone",
    ["resnet18", "resnet34", "mobilenet_v3_small", "efficientnet_b0"],
)
top_k = st.sidebar.slider("Top-K", min_value=1, max_value=10, value=5)

uploaded = st.file_uploader("Upload a Pokemon image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    st.stop()

ckpt_file = Path(checkpoint_path)
if not ckpt_file.exists():
    st.error(f"Checkpoint not found: {ckpt_file}")
    st.stop()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = load_class_names(ckpt_file.parent / "class_names.json")
model = build_model(backbone=backbone, num_classes=len(class_names), pretrained=False)
checkpoint = torch.load(ckpt_file, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

weights = get_weights(backbone, pretrained=True)
preprocess = weights.transforms() if weights is not None else None

image = Image.open(uploaded).convert("RGB")
st.image(image, caption="Input image", use_container_width=True)

with torch.no_grad():
    if preprocess is None:
        tensor = torch.as_tensor(list(image.resize((224, 224)).getdata()), dtype=torch.float32)
        tensor = tensor.view(224, 224, 3).permute(2, 0, 1) / 255.0
    else:
        tensor = preprocess(image)
    batch = tensor.unsqueeze(0).to(device)
    logits = model(batch)
    probs = torch.softmax(logits, dim=1)[0]
    values, indices = torch.topk(probs, k=min(top_k, len(class_names)))

st.subheader("Predictions")
for rank, (score, index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
    st.write(f"{rank}. {class_names[index]}: {score * 100:.2f}%")
