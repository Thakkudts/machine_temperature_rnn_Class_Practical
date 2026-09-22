import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load trained model
model = tf.keras.models.load_model(
    "machine_temperature_rnn.keras"
)

# Load the SAME scalers used during training
X_scaler = joblib.load("X_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")


# Page title
st.title("Machine Temperature Predictor")

st.write(
    "Enter the temperature and vibration values "
    "from the previous two timestamps."
)


# Previous Timestamp 1
st.subheader("Previous Timestamp 1")

temperature_1 = st.number_input(
    "Temperature 1 (°C)",
    value=81.0
)

vibration_1 = st.number_input(
    "Vibration 1",
    value=3.5
)


# Previous Timestamp 2
st.subheader("Previous Timestamp 2")

temperature_2 = st.number_input(
    "Temperature 2 (°C)",
    value=83.0
)

vibration_2 = st.number_input(
    "Vibration 2",
    value=3.6
)


# Prediction button
if st.button("Predict Next Temperature"):

    # Create input sequence
    input_data = np.array([
        [temperature_1, vibration_1],
        [temperature_2, vibration_2]
    ])

    # Scale input using the SAME X_scaler
    input_scaled = X_scaler.transform(input_data)

    # RNN input shape:
    # (samples, time steps, features)
    input_scaled = input_scaled.reshape(
        (1, 2, 2)
    )

    # Predict using trained RNN
    prediction_scaled = model.predict(
        input_scaled,
        verbose=0
    )

    # Convert scaled prediction back to °C
    prediction = y_scaler.inverse_transform(
        prediction_scaled
    )

    predicted_temperature = float(
        prediction[0][0]
    )

    # Display result
    st.success(
        f"Predicted Next Machine Temperature: "
        f"{predicted_temperature:.2f} °C"
    )
