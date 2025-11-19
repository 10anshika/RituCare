import pickle
import pandas as pd
import numpy as np
import os

def load_pcos_model():
    """Load the trained PCOS prediction model and preprocessor."""
    if not os.path.exists('models/pcos_model.pkl') or not os.path.exists('models/preprocessor.pkl'):
        raise FileNotFoundError("PCOS model files not found. Please ensure 'models/pcos_model.pkl' and 'models/preprocessor.pkl' exist.")
    with open('models/pcos_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    return model, preprocessor

def predict_pcos_risk(features_dict):
    """
    Predict PCOS risk based on input features.

    Args:
        features_dict (dict): Dictionary of feature values

    Returns:
        dict: Prediction results with risk level and probability
    """
    try:
        model, preprocessor = load_pcos_model()

        # Convert to DataFrame
        df = pd.DataFrame([features_dict])

        # Convert object columns to numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass

        # Preprocess
        X_processed = preprocessor.transform(df)

        # Predict
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0][1]

        risk_level = "High Risk" if prediction == 1 else "Low Risk"

        return {
            'prediction': int(prediction),
            'risk_level': risk_level,
            'probability': float(probability),
            'confidence': float(max(probability, 1-probability))
        }

    except Exception as e:
        return {'error': str(e)}

def get_feature_importance():
    """Get feature importance from the trained model."""
    try:
        model, preprocessor = load_pcos_model()
        if hasattr(model, 'feature_importances_'):
            # Get feature names from preprocessor
            feature_names = preprocessor.feature_names_in_
            importances = model.feature_importances_
            return dict(zip(feature_names, importances))
        else:
            return {}
    except Exception as e:
        return {'error': str(e)}
