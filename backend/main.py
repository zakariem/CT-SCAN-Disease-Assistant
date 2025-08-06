from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from PIL import Image
import io
import numpy as np

from utils import (
    load_trained_model, preprocess_image, predict, tensor_to_base64,
    GradCAM, SaliencyMap, GuidedBackprop, create_superimposed_image,
    create_probability_chart, get_gemini_explanation,
    DEVICE, DIAGNOSTIC_CLASSES, GATEKEEPER_CLASSES
)

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    print(f"Loading models to device: {DEVICE}")
    models["gatekeeper"] = load_trained_model(
        "MobileNet_V3_Large", len(GATEKEEPER_CLASSES), "models/gatekeeper_model_best.pth"
    )
    models["diagnostic"] = load_trained_model(
        "EfficientNet-B0", len(DIAGNOSTIC_CLASSES), "models/EfficientNet-B0_tuned_best.pth"
    )
    print("Models loaded successfully.")
    yield
    # Clean up models
    models.clear()

app = FastAPI(lifespan=lifespan)

# Allow CORS for frontend communication
origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        request_object_content = await file.read()
        image = Image.open(io.BytesIO(request_object_content))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    processed_tensor = preprocess_image(image)

    # --- Stage 1: Gatekeeper Model ---
    gatekeeper_pred, _ = predict(models["gatekeeper"], processed_tensor, GATEKEEPER_CLASSES)

    if gatekeeper_pred != 'CT_Scan':
        context = f"The initial 'Gatekeeper' model classified the image as '{gatekeeper_pred}', not a 'CT_Scan'. The analysis pipeline was therefore stopped."
        explanation = get_gemini_explanation(context)
        return {
            "gatekeeper_prediction": gatekeeper_pred,
            "diagnostic_prediction": "N/A",
            "message": "Image is not a CT scan. Further analysis halted.",
            "explanation": explanation
        }

    # --- Stage 2: Diagnostic Model ---
    diagnostic_pred, probabilities = predict(models["diagnostic"], processed_tensor, DIAGNOSTIC_CLASSES)

    # --- Stage 3: Generate Visualizations ---
    original_b64 = tensor_to_base64(processed_tensor)
    
    # Get original image in numpy format for superimposing
    original_np = processed_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    original_np = np.clip(original_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)

    # Grad-CAM
    grad_cam_tool = GradCAM(models["diagnostic"], models["diagnostic"].features[-1][0])
    grad_cam_heatmap = grad_cam_tool.generate_heatmap(processed_tensor)
    grad_cam_b64 = create_superimposed_image(grad_cam_heatmap, original_np)

    # Saliency Map
    saliency_tool = SaliencyMap(models["diagnostic"])
    saliency_heatmap = saliency_tool.generate_heatmap(processed_tensor.clone()) # Clone to avoid grad issues
    saliency_b64 = create_superimposed_image(saliency_heatmap, original_np)

    # Guided Backpropagation
    gbp_tool = GuidedBackprop(models["diagnostic"])
    gbp_heatmap = gbp_tool.generate_heatmap(processed_tensor.clone())
    gbp_tool.remove_hooks()
    gbp_b64 = create_superimposed_image(gbp_heatmap, original_np)

    # Probability Chart
    prob_chart_b64 = create_probability_chart(probabilities, DIAGNOSTIC_CLASSES)

    visualizations = {
        "original": original_b64,
        "probability_chart": prob_chart_b64,
        "grad_cam": grad_cam_b64,
        "saliency_map": saliency_b64,
        "guided_backprop": gbp_b64
    }
    
    # --- Stage 4: Get Gemini Explanation ---
    confidence_score = probabilities[np.argmax(probabilities)]
    context = f"""
    - **Gatekeeper Model Result:** Classified as '{gatekeeper_pred}'.
    - **Diagnostic Model Prediction:** '{diagnostic_pred}' with a confidence of {confidence_score:.2%}.
    - **Top 3 Probabilities:** { {k: f'{v:.2%}' for k, v in sorted(zip(DIAGNOSTIC_CLASSES, probabilities), key=lambda item: item[1], reverse=True)[:3]} }.
    - **Visualizations Provided:** Original, Class Probabilities Chart, Grad-CAM, Saliency Map, Guided Backpropagation.
    """
    explanation = get_gemini_explanation(context)

    return {
        "gatekeeper_prediction": gatekeeper_pred,
        "diagnostic_prediction": diagnostic_pred,
        "probabilities": {class_name: f"{prob:.4f}" for class_name, prob in zip(DIAGNOSTIC_CLASSES, probabilities)},
        "visualizations": visualizations,
        "explanation": explanation
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)