import streamlit as st
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model(
    "machine_temperature_rnn.keras"
)

# Page title
st.title("Machine Temperature Predictor")

st.write(
    "Enter the temperature and vibration values "
    "from the previous two timestamps."
)

st.subheader("Previous Timestamp 1")

temperature_1 = st.number_input(
    "Temperature 1 (°C)",
    value=81.0
)

vibration_1 = st.number_input(
    "Vibration 1",
    value=3.5
)

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

    # RNN input shape:
    # (samples, time steps, features)

    input_data = input_data.reshape(
        (1, 2, 2)
    )

    # Prediction
    prediction = model.predict(
        input_data,
        verbose=0
    )

    predicted_temperature = float(
        prediction[0][0]
    )

    st.success(
        f"Predicted Next Machine Temperature: "
        f"{predicted_temperature:.2f} °C"
    )
