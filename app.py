import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load trained model
model = tf.keras.models.load_model(
    "machine_temperature_rnn.keras"
)

# -------------------------------------------------
# Recreate the SAME training data
# -------------------------------------------------

temperature = np.array([
    60, 62, 61, 64, 65,
    67, 66, 69, 70, 72,
    71, 74, 75, 77, 76,
    79, 80, 82, 81, 83
], dtype=float)

vibration = np.array([
    2.1, 2.3, 2.2, 2.5, 2.6,
    2.7, 2.5, 2.8, 2.9, 3.0,
    2.9, 3.1, 3.2, 3.3, 3.1,
    3.4, 3.5, 3.6, 3.4, 3.6
], dtype=float)

# Create the same feature data used during training
features = np.column_stack((temperature, vibration))

# -------------------------------------------------
# Recreate X scaler
# -------------------------------------------------

X_scaler = MinMaxScaler()

X_scaler.fit(features)

# -------------------------------------------------
# Recreate y scaler
# -------------------------------------------------

y_scaler = MinMaxScaler()

y_scaler.fit(temperature.reshape(-1, 1))


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------

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


# -------------------------------------------------
# Prediction
# -------------------------------------------------

if st.button("Predict Next Temperature"):

    # Raw input
    input_data = np.array([
        [temperature_1, vibration_1],
        [temperature_2, vibration_2]
    ])

    # Scale input
    input_scaled = X_scaler.transform(input_data)

    # Reshape for RNN
    input_scaled = input_scaled.reshape(
        (1, 2, 2)
    )

    # Model prediction
    prediction_scaled = model.predict(
        input_scaled,
        verbose=0
    )

    # Convert back to °C
    prediction = y_scaler.inverse_transform(
        prediction_scaled
    )

    predicted_temperature = float(
        prediction[0][0]
    )

    st.success(
        f"Predicted Next Machine Temperature: "
        f"{predicted_temperature:.2f} °C"
    )
