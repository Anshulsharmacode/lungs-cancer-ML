import numpy as np
from PIL import Image
from skimage import color, filters
from skimage.feature import graycomatrix, graycoprops
from fastapi import FastAPI, UploadFile, File, HTTPException
from sklearn.ensemble import RandomForestClassifier
from io import BytesIO
from sklearn.model_selection import train_test_split

# Initialize FastAPI
app = FastAPI()

# Tumor types mapping
tumor_types = {
    'adenocarcinoma': 'Non-Small Cell Lung Cancer (NSCLC) - Adenocarcinoma',
    'squamous_cell': 'Non-Small Cell Lung Cancer (NSCLC) - Squamous Cell Carcinoma',
    'large_cell': 'Non-Small Cell Lung Cancer (NSCLC) - Large Cell Carcinoma',
    'small_cell': 'Small Cell Lung Cancer (SCLC)',
    'metastatic': 'Secondary Lung Tumors - Metastatic',
    'hamartoma': 'Benign Lung Tumors - Hamartomas',
    'pulmonary_adenoma': 'Benign Lung Tumors - Pulmonary Adenomas'
}

# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

def extract_image_features(image: np.array):
    """Extract features from a given image array."""
    if image.shape[-1] == 4:  # Handle RGBA images
        image = color.rgba2rgb(image)

    # Convert to grayscale and normalize
    image_gray = color.rgb2gray(image)
    image_normalized = (image_gray - np.min(image_gray)) / (np.max(image_gray) - np.min(image_gray))

    # Extract intensity features
    mean_intensity = np.mean(image_normalized)
    std_intensity = np.std(image_normalized)
    cv = std_intensity / mean_intensity if mean_intensity > 0 else 0
    
    # Extract edge strength using Sobel filter
    edges_sobel = filters.sobel(image_normalized)
    avg_edge_strength = np.mean(edges_sobel)
    
    # Extract GLCM texture features
    glcm = graycomatrix((image_normalized * 255).astype(np.uint8), distances=[1], angles=[0], 
                        levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    
    return [mean_intensity, std_intensity, cv, avg_edge_strength, contrast, 
            dissimilarity, homogeneity, energy, correlation, asm]

@app.get("/")
def read_root():
    return {"message": "Welcome to the Tumor Prediction API! Use the /predict-tumor/ endpoint to upload images."}

#whatsapp dhkturn me

@app.post("/predict-tumor/")
async def predict_tumor(file: UploadFile = File(...)):
    try:
        # Read the image file
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_array = np.array(image)
        
        # Extract features from the image
        image_features = extract_image_features(image_array)
        
        # Reshape the features for prediction
        features = np.array(image_features).reshape(1, -1)
        
        # Predict the tumor type
        prediction = model.predict(features)
        
        # Return the result
        predicted_label = prediction[0]
        tumor_type = tumor_types.get(predicted_label, "Unknown Tumor Type")
        
        return {"tumor_type": tumor_type, "predicted_label": predicted_label}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred while processing the image: {e}")

# Training the model with dummy data for demonstration
def train_model():
    # Generate dummy data for training
    # In practice, replace this with your real image feature extraction
    X_dummy = np.random.rand(100, 10)  # 100 samples, 10 features
    y_dummy = np.random.choice(list(tumor_types.keys()), size=100)  # Random labels from tumor_types

    # Split the dummy data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_dummy, test_size=0.2, random_state=42)

    # Fit the model on dummy data
    model.fit(X_train, y_train)

# Call the training function when the application starts
train_model()

# To run this FastAPI application, use the command:
# uvicorn main:app --host 0.0.0.0 --port 5000
