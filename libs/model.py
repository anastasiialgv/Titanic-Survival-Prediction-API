import numpy as np
import pickle
from pathlib import Path

def predict(features: list, ml_model: Path):
    input_data = np.array([features], dtype=np.float64) 
    
    try:
        with open(ml_model, 'rb') as file:
            model = pickle.load(file)
    except Exception as e:
        raise IOError(f"Error loading Scikit-learn model (pickle): {e}")

    prediction_array = model.predict_proba(input_data)
    
    return prediction_array[0][1] 

def train(features: list, ml_model: Path):
    try:
        with open(ml_model, 'rb') as file:
            model = pickle.load(file)
            
        with open(ml_model, 'wb') as file:
            pickle.dump(model, file)
            
        return True
    except Exception as e:
        raise IOError(f"Error during model saving/access (pickle): {e}")