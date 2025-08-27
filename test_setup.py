#!/usr/bin/env python3
"""
Test script to verify your uploaded model and scaler work correctly.
Run this after uploading your files from Colab.
"""

import joblib
import numpy as np
from pathlib import Path

def test_uploaded_model():
    """Test the uploaded model and scaler."""
    
    print("🧪 Testing uploaded model and scaler...")
    print("=" * 50)
    
    # Check if files exist
    model_path = Path("app/models/xgb_model.joblib")
    scaler_path = Path("app/models/minmax_scaler.joblib")
    
    if not model_path.exists():
        print("❌ Model file not found: app/models/xgb_model.joblib")
        print("   Please upload your model file from Colab to this location.")
        return False
    
    if not scaler_path.exists():
        print("❌ Scaler file not found: app/models/minmax_scaler.joblib")
        print("   Please upload your scaler file from Colab to this location.")
        return False
    
    print("✅ Both model and scaler files found!")
    
    try:
        # Load model and scaler
        print("\n📥 Loading model and scaler...")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        print(f"✅ Model loaded: {type(model).__name__}")
        print(f"✅ Scaler loaded: {type(scaler).__name__}")
        
        # Test with sample data
        print("\n🧪 Testing with sample data...")
        test_data = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
        
        # Scale the data
        scaled_data = scaler.transform(test_data)
        print(f"✅ Data scaled successfully. Shape: {scaled_data.shape}")
        
        # Make prediction
        prediction = model.predict(scaled_data)
        print(f"✅ Prediction successful: {prediction[0]}")
        
        # Test probability prediction
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(scaled_data)
            print(f"✅ Probability prediction successful: {probabilities[0]}")
        else:
            print("⚠️  Model doesn't support probability prediction")
        
        print("\n🎉 All tests passed! Your model is ready to use.")
        print("\nNext steps:")
        print("1. Start the FastAPI server: python app/main.py")
        print("2. Open your browser to: http://localhost:8000")
        print("3. Test the prediction with the web interface")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure your model is an XGBoost classifier")
        print("2. Ensure your model expects 13 features in the correct order")
        print("3. Check that your scaler was fitted on the same feature set")
        return False

if __name__ == "__main__":
    test_uploaded_model()

