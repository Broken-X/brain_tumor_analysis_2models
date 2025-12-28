import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
MODEL_DIR = "brain_tumor_models"
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# --- 2. STABLE GRAD-CAM UTILITY ---

def disable_inplace_ops(model: nn.Module) -> None:
    """Disable inplace ops (e.g., ReLU(inplace=True)) that can break Grad-CAM hooks on some models."""
    for m in model.modules():
        if hasattr(m, 'inplace') and isinstance(getattr(m, 'inplace'), bool):
            m.inplace = False

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # IMPORTANT:
        # Do NOT use module backward hooks (register_backward_hook / register_full_backward_hook)
        # for Grad-CAM on torchvision models like DenseNet. Those hooks wrap backward in a
        # BackwardHookFunction and can trigger the exact "view + inplace" error you're seeing.
        #
        # Instead, capture gradients with a *tensor hook* attached to the activation tensor.
        self._fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        # Save activations safely (clone avoids view/inplace issues)
        self.activations = output.clone().detach()

        # Attach a tensor hook to grab gradients during backprop.
        # This avoids BackwardHookFunctionBackward entirely.
        def _tensor_grad_hook(grad):
            self.gradients = grad.clone().detach()

        # Only register if gradients will flow.
        if isinstance(output, torch.Tensor) and output.requires_grad:
            output.register_hook(_tensor_grad_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[0, class_idx]
        target.backward()

        # Weight channels by gradients (Global Average Pooling)
        # self.gradients must have 4 dimensions (B, C, H, W)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().cpu().numpy()
        
        # ReLU and Normalization
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        denom = cam.max() - cam.min()
        return (cam - cam.min()) / (denom if denom != 0 else 1e-7)

    def close(self):
        # Remove hooks to prevent accumulation on repeated Streamlit runs
        try:
            if getattr(self, '_fwd_handle', None) is not None:
                self._fwd_handle.remove()
        except Exception:
            pass

def apply_heatmap(img_pil, mask):
    # Convert mask to heatmap overlay using OpenCV
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_np = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE)))
    superimposed = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    return superimposed

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_all_models():    
    # EfficientNet B0 Architecture
    m_eff = models.efficientnet_b0()
    m_eff.classifier[1] = nn.Linear(m_eff.classifier[1].in_features, 4)
    disable_inplace_ops(m_eff)
    
    # DenseNet 121 Architecture (Fixed Syntax)
    m_den = models.densenet121()
    m_den.classifier = nn.Linear(m_den.classifier.in_features, 4)
    disable_inplace_ops(m_den)
    
    # MAPPING: Target final convolutional features for Grad-CAM
    mapping = {
        'efficientnet_b0': (m_eff, m_eff.features),
        'densenet121': (m_den, m_den.features) 
    }

    loaded_models = {}
    for name, (model, target) in mapping.items():
        path = os.path.join(MODEL_DIR, f"{name}_brain_tumor.pth")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(DEVICE).eval()
            loaded_models[name] = (model, target)
    return loaded_models

# --- 4. STREAMLIT INTERFACE ---
st.set_page_config(layout="wide", page_title="Brain Tumor Analysis")
st.title("🧠 Brain Tumor MRI: Side-by-Side Model Analysis")

uploaded_file = st.file_uploader("Upload an MRI scan...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_file:
    for uploaded_file in uploaded_file:
        st.markdown(f"### 📄 {uploaded_file.name}")
        image = Image.open(uploaded_file).convert('RGB')
        
        # Preprocessing
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        models_dict = load_all_models()
        
        if models_dict:
            cols = st.columns(len(models_dict) + 1)

            # Show original image in the first column
            with cols[0]:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)

            for i, (name, (model, target_layer)) in enumerate(models_dict.items()):
                with cols[i + 1]:
                    st.subheader(f"Model: {name.upper()}")
                    
                    try:
                        # Grad-CAM and Inference
                        gcam = GradCAM(model, target_layer)
                        output = model(input_tensor)
                        prob = torch.nn.functional.softmax(output, dim=1)
                        conf, pred_idx = torch.max(prob, 1)
                        
                        # Generate visual heatmap
                        mask = gcam.generate(input_tensor, pred_idx.item())
                        heatmap_img = apply_heatmap(image, mask)
                        
                        # UI display
                        st.image(heatmap_img, use_container_width=True)
                        st.metric("Prediction", CLASS_NAMES[pred_idx.item()])
                        st.write(f"Confidence: **{conf.item():.2%}**")

                        # Clean up hooks (important in Streamlit reruns)
                        gcam.close()
                    except Exception as e:
                        try:
                            gcam.close()
                        except Exception:
                            pass
                        st.error(f"Error in {name}: {e}")
        else:
            st.error("No saved models found. Check your 'brain_tumor_models' folder.")