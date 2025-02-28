import joblib
import numpy as np
import pandas as pd

def load_model(model_path='ml_pipeline_model.pkl'):
    return joblib.load(model_path)

def get_input_features(feature_names):
    try:
        user_input = input("Enter feature values separated by spaces (or press enter to exit): ")
        if user_input.strip() == "":
            return None
        features = list(map(float, user_input.split()))
        if len(features) != len(feature_names):
            print(f"Error: Expected {len(feature_names)} values, but got {len(features)}.")
            return get_input_features(feature_names)
        return pd.DataFrame([features], columns=feature_names)
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return get_input_features(feature_names)

def main():
    model = load_model()
    print("Model loaded successfully. Enter feature values for prediction.")
    feature_names = ['ia', 'ib', 'ic', 'va', 'vb', 'vc']  # Ensure these match your model's training data
    
    while True:
        features = get_input_features(feature_names)
        if features is None:
            print("Exiting...")
            break
        prediction = model.predict(features)
        print("Predicted class:", prediction[0])

if __name__ == "__main__":
    main()