from fastapi import FastAPI, UploadFile, HTTPException, Request, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import io
import numpy as np

app = FastAPI()

# Load the machine learning model
model = load_model("nidarshan_model.h5")

# Define the labels corresponding to your model's output
labels = [
    "Acne and Rosacea Photos",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Atopic Dermatitis Photos",
    "Bullous Disease Photos",
    "Cellulitis Impetigo and other Bacterial Infections",
    "Eczema Photos",
    "Exanthems and Drug Eruptions",
    "Hair Loss Photos Alopecia and other Hair Diseases",
    "Herpes HPV and other STDs Photos",
    "Light Diseases and Disorders of Pigmentation",
    "Lupus and other Connective Tissue diseases",
    "Melanoma Skin Cancer Nevi and Moles",
    "Nail Fungus and other Nail Disease",
    "Poison Ivy Photos and other Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Scabies Lyme Disease and other Infestations and Bites",
    "Seborrheic Keratoses and other Benign Tumors",
    "Systemic Disease",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Urticaria Hives",
    "Vascular Tumors",
    "Vasculitis Photos",
    "Warts Molluscum and other Viral Infections"
]

def preprocess_image(image_bytes):
    try:
        # Open the image using Pillow (PIL)
        img = Image.open(io.BytesIO(image_bytes))

        # Resize the image to the expected input size
        img = img.resize((299, 299))

        # Preprocess the image for your model
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize the pixel values

        return img_array.reshape((1, 299, 299, 3))
    except Exception as e:
        # Handle exceptions appropriately based on your requirements
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/")
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        processed_image = preprocess_image(contents)

        # Make predictions using the loaded model
        prediction = model.predict(processed_image)

        # Get the indices of the top 5 predictions
        top_indices = np.argsort(prediction[0])[::-1][:5]

        # Create a list of dictionaries with disease and probability
        top_predictions = [{"disease": labels[idx], "probability": float(prediction[0][idx])} for idx in top_indices]

        # Include the top predictions in the JSON response
        return JSONResponse(content={"top_predictions": top_predictions})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)