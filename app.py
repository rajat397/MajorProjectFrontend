import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

# Load TorchScript model (.pt)
@st.cache_resource
def load_model(model_path="deepfake_model.pt"):
    model = torch.jit.load(model_path, map_location=torch.device("cpu"))
    model.eval()
    return model

# Prediction function
def predict(model, image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),   # Adjust to match training size
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    label = "Real" if pred_class == 1 else "Deepfake"
    return label, confidence

# Streamlit UI
st.title("🕵️‍♂️ Deepfake Image Detector")
uploaded_file = st.file_uploader("Upload an image (JPG/PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        model = load_model()
        label, confidence = predict(model, image)

    if label == "Deepfake":
        st.error(f"⚠️ Prediction: **{label}** (Confidence: {confidence:.2f})")
    else:
        st.success(f"✅ Prediction: **{label}** (Confidence: {confidence:.2f})")
