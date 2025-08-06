import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import io
import base64
import os
from dotenv import load_dotenv
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Found from notebook output
DIAGNOSTIC_CLASSES = [
    'Colon_Cancer', 'Colon_Non_Cancer', 'Kidney_Cyst', 'Kidney_Normal',
    'Kidney_Stone', 'Kidney_Tumor', 'Liver_Hepatic_Steatosis', 'Liver_Healthy'
]
GATEKEEPER_CLASSES = ['CT_Scan', 'X_Ray', 'Unknown']

# --- Model Loading ---
def get_model(model_name: str, num_classes: int, pretrained: bool = True):
    weights = "IMAGENET1K_V1" if pretrained else None
    model = None
    if model_name == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "MobileNet_V3_Large":
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Model '{model_name}' not supported.")
    return model

def load_trained_model(model_name, num_classes, path):
    model = get_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# --- Image Preprocessing ---
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    return val_test_transform(image).unsqueeze(0).to(DEVICE)

def tensor_to_base64(tensor):
    img_np = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np * std + mean, 0, 1)
    pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- Prediction Functions ---
def predict(model, processed_tensor, class_names):
    with torch.no_grad():
        outputs = model(processed_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        prediction_idx = torch.argmax(probabilities).item()
        prediction_name = class_names[prediction_idx]
    return prediction_name, probabilities.cpu().numpy()

# --- Visualization Tools ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()
    
    def hook_layers(self):
        def forward_hook(module, input, output): self.activations = output
        def backward_hook(module, grad_in, grad_out): self.gradients = grad_out[0]
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        class_idx = output.argmax(dim=1).item()
        output[0, class_idx].backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        return heatmap.detach().cpu().numpy()

class SaliencyMap:
    def __init__(self, model):
        self.model = model

    def generate_heatmap(self, input_tensor):
        input_tensor.requires_grad_()
        self.model.zero_grad()
        output = self.model(input_tensor)
        class_idx = output.argmax(dim=1).item()
        output[0, class_idx].backward()
        
        saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
        saliency = saliency.squeeze().cpu().numpy()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        return saliency

class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.hook_relu()

    def hook_relu(self):
        def guided_relu_hook(module, grad_in, grad_out):
            return (torch.clamp(grad_in[0], min=0.0),)
        
        for module in self.model.named_modules():
            if isinstance(module[1], nn.ReLU):
                self.hooks.append(module[1].register_backward_hook(guided_relu_hook))
    
    def generate_heatmap(self, input_tensor):
        input_tensor.requires_grad_()
        self.model.zero_grad()
        output = self.model(input_tensor)
        class_idx = output.argmax(dim=1).item()
        output[0, class_idx].backward()

        guided_grads = input_tensor.grad.data.squeeze().cpu().numpy()
        guided_grads = np.transpose(guided_grads, (1, 2, 0))
        guided_grads = (guided_grads - guided_grads.min()) / (guided_grads.max() - guided_grads.min())
        return guided_grads

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

def create_superimposed_image(heatmap, original_img_np):
    heatmap_resized = cv2.resize(heatmap, (original_img_np.shape[1], original_img_np.shape[0]))
    if heatmap_resized.ndim == 2: # For GradCAM and Saliency
      heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
      heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
      superimposed = np.clip(heatmap_color * 0.4 + original_img_np * 0.6, 0, 1)
    else: # For Guided Backprop
      superimposed = heatmap
    
    pil_img = Image.fromarray((superimposed * 255).astype(np.uint8))
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def create_probability_chart(probabilities, class_names):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a DataFrame for easier plotting
    prob_df = pd.DataFrame({
        'Class': class_names,
        'Probability': probabilities
    }).sort_values(by='Probability', ascending=True)

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(prob_df)))
    ax.barh(prob_df['Class'], prob_df['Probability'], color=colors)

    ax.set_title('Prediction Probabilities', fontsize=16, weight='bold')
    ax.set_xlabel('Probability', fontsize=12)
    ax.tick_params(axis='y', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# --- Gemini AI Integration ---
def get_gemini_explanation(context):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        You are an expert radiology AI assistant. Your role is to explain the results from a deep learning model to a radiologist in a clear, concise, and professional manner. Do not diagnose. Explain the model's findings.

        **Analysis Context:**
        {context}

        **Your Task:**
        Provide a structured explanation based on the context above. Use Markdown for formatting. Structure your response as follows:

        ### 1. Summary of Findings
        Briefly state the model's primary prediction and its confidence level.

        ### 2. Interpretation of Probabilities
        Explain the probability distribution across all classes. Mention the top predictions and why the model is confident or uncertain. Reference the "Class Probabilities Chart".

        ### 3. Visualization Analysis
        Explain what each provided visualization signifies in simple terms for a radiologist.
        - **Grad-CAM:** Explain that the "hot" areas are what the model focused on to make its decision.
        - **Saliency Map:** Describe this as highlighting the most sensitive pixels that influenced the outcome.
        - **Guided Backpropagation:** Explain this shows fine-grained features (like edges and textures) the model used.
        
        ### 4. Conclusion & Disclaimer
        Provide a concluding thought and a clear disclaimer that these are AI-generated results for assistance and are not a substitute for a final diagnosis by a qualified radiologist.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error connecting to Gemini AI: {e}\nPlease check your API key and network connection."