import numpy as np
import tensorflow as tf
import joblib

SEQ_LEN = 10
MODEL_PATH = "aqi_lstm_attention_model.h5"
SCALER_PATH = "aqi_scaler.save"

# Replace with your real feature names (keep in same order as in training)
feature_names = [
     'NO2', 'CO2', 'SO2', 'dust'
]

# -------------------------------
# 1. LOAD SCALER & MODEL
# -------------------------------

scaler = joblib.load(SCALER_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------------
# 2. PREPARE INPUT
# -------------------------------

# Your single test row as a list:
sample_input = [
    93.94,61.02,1.37,12.42
]

print(len(sample_input))

# replace None with median value (we'll use zeros because scaler will fix scale)
median_values = np.zeros_like(sample_input)
for i in range(len(sample_input)):
    if sample_input[i] is None:
        sample_input[i] = median_values[i]

# Convert to array and scale
sample_input_arr = np.array(sample_input).reshape(1, -1)
sample_input_scaled = scaler.transform(sample_input_arr)

# Repeat same vector SEQ_LEN times to form sequence
X_test_seq = np.repeat(sample_input_scaled.reshape(1, -1), SEQ_LEN, axis=0)
X_test_seq = X_test_seq.reshape(1, SEQ_LEN, -1)

# -------------------------------
# 3. PREDICT
# -------------------------------

pred = model.predict(X_test_seq)
pred_aqi = pred[0][0]
print("Predicted AQI value:", pred_aqi)
