from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import os
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import traceback
import warnings

warnings.filterwarnings('ignore')  # Suppress warnings, especially from SHAP/LIME if they are verbose

# Initialize FastAPI app
app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

# Add CORS middleware
# This is crucial for your frontend (running in a browser) to communicate with this backend API.
# In production, 'allow_origins' should be restricted to your frontend's actual domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE)
    allow_headers=["*"],  # Allows all headers
)

# Mount static files (e.g., your index.html, CSS, JS)
# When a request comes to /static/some_file.html, it will look in the 'static' directory.
app.mount("/static", StaticFiles(directory="static"), name="static")


# Define the input data model using Pydantic
# This ensures that incoming JSON data has the correct fields and types.
class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int  # chest pain type
    trestbps: int  # resting blood pressure
    chol: int  # serum cholesterol
    fbs: int  # fasting blood sugar (fasting blood glucose > 120 mg/dl (1 = true; 0 = false))
    restecg: int  # resting electrocardiographic results
    thalach: int  # maximum heart rate achieved
    exang: int  # exercise induced angina (1 = yes; 0 = no)
    oldpeak: float  # ST depression induced by exercise relative to rest
    slope: int  # the slope of the peak exercise ST segment
    ca: int  # number of major vessels (0-3) colored by flourosopy
    thal: int  # thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)


# Global variables to store the loaded model and explainers
# They are initialized once on startup to avoid reloading for every request.
model = None
lime_explainer = None
shap_explainer = None
X_train_sample_df = None  # Store the sample DataFrame for feature names and context

# Feature names for explanations, derived from the training data sample
feature_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]


def load_models():
    """
    Load the trained model and X_train_sample.
    This function is called once at application startup.
    """
    global model, X_train_sample_df

    # Define candidate paths for the model file (handles common OS/browser renaming)
    model_candidate_paths = [
        Path("app/models/xgb_model.joblib"),
        Path("app/models/xgb_model (1).joblib"),  # For cases where browsers rename downloads
        Path("app/models/xgb_model(1).joblib"),
    ]
    model_path = next((p for p in model_candidate_paths if p.exists()), None)

    if model_path is None:
        raise FileNotFoundError(
            "XGBoost model not found. Expected one of: xgb_model.joblib, xgb_model (1).joblib, xgb_model(1).joblib in app/models/."
        )

    # Load the XGBoost model
    model = joblib.load(model_path)

    # Define candidate paths for the training data sample
    sample_data_candidate_paths = [
        Path("app/models/X_train_sample.joblib"),
        Path("app/models/X_train_sample (1).joblib"),
        Path("app/models/X_train_sample(1).joblib"),
    ]
    sample_data_path = next((p for p in sample_data_candidate_paths if p.exists()), None)

    if sample_data_path is None:
        # If the sample is not found, we'll try to proceed but LIME might fail without proper data.
        print("⚠️ Warning: X_train_sample.joblib not found. LIME explainer might not work correctly.")
        # We can create a dummy DataFrame if absolutely necessary for LIME initialization
        # but it's always better to use real training data.
        # For now, we'll raise an error to ensure the correct file is provided.
        raise FileNotFoundError(
            "Training data sample (X_train_sample.joblib) not found. This is crucial for LIME initialization."
        )

    X_train_sample_df = joblib.load(sample_data_path)
    # Ensure numeric dtypes only and handle missing values
    X_train_sample_df = X_train_sample_df.copy()
    for col in X_train_sample_df.columns:
        X_train_sample_df[col] = pd.to_numeric(X_train_sample_df[col], errors='coerce')
        median_value = X_train_sample_df[col].median()
        if pd.isna(median_value):
            median_value = 0.0
        X_train_sample_df[col] = X_train_sample_df[col].fillna(median_value)
    # Add tiny noise to columns with zero variance to avoid LIME issues
    std_series = X_train_sample_df.std(numeric_only=True)
    zero_std_cols = [c for c in X_train_sample_df.columns if std_series.get(c, 1.0) == 0]
    if zero_std_cols:
        eps = 1e-6
        noise = np.random.default_rng(123).normal(0, eps, size=(len(X_train_sample_df), len(zero_std_cols)))
        X_train_sample_df.loc[:, zero_std_cols] = X_train_sample_df.loc[:, zero_std_cols].values + noise

    # Ensure feature_names matches the order in X_train_sample_df
    global feature_names
    feature_names = X_train_sample_df.columns.tolist()

    print(f"Loaded model from: {model_path}")
    print(f"Loaded X_train_sample from: {sample_data_path}")
    print(f"Feature names for explainers: {feature_names}")

    return model


def initialize_explainers():
    """
    Initialize LIME and SHAP explainers using the loaded model and training data sample.
    This function is called once at application startup.
    """
    global lime_explainer, shap_explainer

    if model is None or X_train_sample_df is None:
        raise ValueError("Model and training data sample must be loaded before initializing explainers.")

    # --- LIME Explainer Initialization ---
    # Convert DataFrame to NumPy array for LIME explainer's training_data
    # Use the loaded X_train_sample_df to ensure correct feature order
    training_data_for_lime = X_train_sample_df.values

    # Define categorical feature indices based on the actual feature_names list
    # These indices MUST match the positions in your feature_names list
    categorical_features_indices = [
        feature_names.index('sex'),
        feature_names.index('cp'),
        feature_names.index('fbs'),
        feature_names.index('restecg'),
        feature_names.index('exang'),
        feature_names.index('slope'),
        feature_names.index('ca'),
        feature_names.index('thal')
    ]

    lime_explainer = LimeTabularExplainer(
        training_data_for_lime,
        feature_names=feature_names,
        class_names=['No Disease', 'Disease'],
        mode='classification',
        categorical_features=categorical_features_indices,
        discretize_continuous=False,
        sample_around_instance=True,
        random_state=42
    )

    # --- SHAP Explainer Initialization ---
    # SHAP TreeExplainer is optimized for tree-based models like XGBoost.
    shap_explainer = shap.TreeExplainer(model)

    print("✅ LIME and SHAP explainers initialized with real data.")


def preprocess_input(input_data: HeartDiseaseInput):
    """
    Prepare raw feature array for the model.
    Since your XGBoost model does not use scaling, this function only converts
    the Pydantic model input into a NumPy array in the correct feature order.
    """
    features_array = np.array([[
        input_data.age, input_data.sex, input_data.cp, input_data.trestbps,
        input_data.chol, input_data.fbs, input_data.restecg, input_data.thalach,
        input_data.exang, input_data.oldpeak, input_data.slope, input_data.ca,
        input_data.thal
    ]])

    return features_array


# FastAPI event handler to run functions on application startup
@app.on_event("startup")
async def startup_event():
    """
    This function runs when the FastAPI application starts up.
    It loads the ML model and initializes the LIME and SHAP explainers.
    """
    try:
        load_models()
        initialize_explainers()
        print("✅ Models and explainers loaded successfully (startup_event).")
    except Exception as e:
        print(f"❌ Error during startup: {str(e)}")
        # If essential components fail to load, raise an exception to prevent app from starting
        raise HTTPException(status_code=500, detail=f"Startup failed: {str(e)}")


# --- API Endpoints ---

@app.post("/predict")
async def predict_heart_disease(input_data: HeartDiseaseInput):
    """
    Endpoint to get heart disease prediction.
    Takes user input, preprocesses it, and returns the model's prediction
    and associated probabilities.
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded. Please check server logs.")

        # Preprocess input data into a NumPy array
        features = preprocess_input(input_data)

        # Make prediction and get probabilities
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        # Return the prediction, probabilities, and a human-readable message
        return {
            "prediction": int(prediction),
            "probabilities": {
                "no_disease": float(probabilities[0]),
                "disease": float(probabilities[1])
            },
            "message": "Heart disease detected" if prediction == 1 else "No heart disease detected",
            "confidence": float(max(probabilities))  # Confidence is the highest probability
        }

    except Exception as e:
        # Catch any exceptions during prediction and return a 500 error
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/explain/lime")
async def explain_lime(input_data: HeartDiseaseInput):
    try:
        if lime_explainer is None:
            raise HTTPException(status_code=500, detail="LIME explainer not loaded.")

        # Preprocess
        features = preprocess_input(input_data)
        predicted_label = int(model.predict(features)[0])

        # Generate LIME explanation
        explanation = lime_explainer.explain_instance(
            features[0],
            model.predict_proba,
            num_features=len(feature_names),  # return all features
            labels=[0, 1],  # ✅ request explanations for both classes
            distance_metric='euclidean'
        )

        # Get contributions for the predicted class
        feature_contributions = [
            {"feature": feat_desc, "weight": float(contribution)}
            for feat_desc, contribution in explanation.as_list(label=predicted_label)
        ]

        return {
            "feature_contributions": feature_contributions,
            "prediction": predicted_label
        }

    except Exception as e:
        print("❌ LIME explanation failed:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"LIME explanation error: {str(e)}")


@app.post("/explain/shap")
async def explain_shap(input_data: HeartDiseaseInput):
    """
    Endpoint to get SHAP explanation for a single prediction.
    Returns SHAP values, the base value, and feature names.
    """
    try:
        if shap_explainer is None:
            raise HTTPException(status_code=500, detail="SHAP explainer not loaded. Please check server logs.")

        # Preprocess input data
        features = preprocess_input(input_data)

        try:
            # ✅ New SHAP API (preferred)
            shap_result = shap_explainer(features)
            shap_values = shap_result.values[0]  # SHAP values for this instance
            base_value = float(shap_result.base_values[0])
        except Exception:
            # ✅ Old SHAP API fallback
            shap_values_raw = shap_explainer.shap_values(features)
            if isinstance(shap_values_raw, list):  # binary classification
                shap_values = shap_values_raw[1][0]
                base_value = float(shap_explainer.expected_value[1])
            else:
                shap_values = shap_values_raw[0]
                base_value = float(shap_explainer.expected_value)

        current_prediction = int(model.predict(features)[0])

        return {
            "shap_values": shap_values.tolist(),
            "base_value": base_value,
            "feature_names": feature_names,
            "prediction": current_prediction
        }

    except Exception as e:
        print("❌ SHAP explanation failed:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"SHAP explanation error: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API status and component loading.
    """
    model_status = "loaded" if model is not None else "not loaded"
    lime_status = "loaded" if lime_explainer is not None else "not loaded"
    shap_status = "loaded" if shap_explainer is not None else "not loaded"

    return {
        "status": "healthy",
        "message": "Heart Disease Prediction API is running",
        "models": {
            "xgboost_model": model_status,
            "lime_explainer": lime_status,
            "shap_explainer": shap_status
        }
    }


@app.get("/")
async def read_index():
    """
    Serves the main HTML frontend application.
    """
    # Use os.path.join for cross-platform compatibility
    index_path = os.path.join("static", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found in static folder.")
    return FileResponse(index_path)


@app.get("/model-info")
async def get_model_info():
    """
    Provides information about the loaded model and explainers.
    Useful for frontend to verify backend setup.
    """
    if model is None:
        return {"status": "Model not loaded", "message": "Please check server logs"}

    return {
        "model_type": type(model).__name__,
        "model_loaded": True,
        "lime_loaded": lime_explainer is not None,
        "shap_loaded": shap_explainer is not None,
        "features": feature_names,
        "feature_count": len(feature_names)
    }


# Entry point for running the Uvicorn server directly from this script
if __name__ == "__main__":
    import uvicorn

    # Allow host and port to be configured via environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port_str = os.getenv("PORT", "8000")
    try:
        port = int(port_str)
    except ValueError:
        port = 8000  # Default to 8000 if PORT env var is invalid
    uvicorn.run(app, host=host, port=port)
