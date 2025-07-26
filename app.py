import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

# Set page config
st.set_page_config(page_title="Crop Stage Prediction", layout="wide")

st.title("Crop Stage Prediction using LSTM")

# Load the trained model
@st.cache_resource
def load_model():
    model_path = 'lstm_crop_stage_model.h5'
    if not os.path.exists(model_path):
         # Adjust path for deployment environment if needed
         model_path = os.path.join(os.path.dirname(__file__), model_path)
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the scalers
@st.cache_resource
def load_scalers():
    try:
        scaler_X = joblib.load('scaler_X.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
        return scaler_X, scaler_y
    except Exception as e:
        st.error(f"Error loading scalers: {e}")
        return None

model = load_model()
scaler_X, scaler_y = load_scalers()

if model and scaler_X and scaler_y:
    st.sidebar.header("Input Features (Latest Week)")

    # Get feature names from the scaler
    feature_names = scaler_X.feature_names_in_

    # Create input fields for each feature
    input_data = {}
    for feature in feature_names:
        # You might want to customize the input type based on the feature (e.g., number_input, selectbox)
        input_data[feature] = st.sidebar.number_input(f"Enter {feature}", value=0.0) # Default value 0.0

    # Add a button to make prediction
    if st.sidebar.button("Predict Crop Stage"):
        try:
            # Prepare input data for prediction
            input_df = pd.DataFrame([input_data])

            # Scale the input data
            input_scaled = scaler_X.transform(input_df)

            # Reshape for LSTM (batch_size, sequence_length, num_features)
            # Assuming a sequence length of 1 (predicting based on the latest week's data)
            # Note: For true sequence prediction, you'd need historical data as input.
            # This example uses the latest week as a single-step prediction input.
            input_reshaped = input_scaled.reshape(1, 1, input_scaled.shape[1])


            # Make prediction
            prediction_scaled = model.predict(input_reshaped)

            # Inverse transform the prediction
            prediction_original = scaler_y.inverse_transform(prediction_scaled)

            st.subheader("Predicted Crop Stage")
            st.write(f"The predicted crop stage is: {prediction_original[0][0]:.2f}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

else:
    st.warning("Model or scalers could not be loaded. Please check the file paths.")

st.markdown("---")
st.write("Note: This is a simplified prediction based on single week's input. For more accurate predictions, a sequence of historical data is needed.")
