import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    Attention,
    LayerNormalization,
    RepeatVector,
    TimeDistributed,
    Concatenate,
    Conv1D,
    MaxPooling1D,
    Flatten,
    GlobalMaxPooling1D,
    Reshape,
    BatchNormalization,
    Layer
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# -------------------------------
# CONFIG
# -------------------------------

SEQ_LEN = 10         # number of time steps to look back
BATCH_SIZE = 32
EPOCHS = 200
PATIENCE = 20
MODEL_PATH = "aqi_cnn_lstm_attention_model.keras"
SCALER_PATH = "aqi_scaler.save"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Custom layers for indexing operations
class LastTimestepLayer(Layer):
    def call(self, inputs):
        return inputs[:, -1:, :]
    
    def get_config(self):
        return super().get_config()

class LastOutputLayer(Layer):
    def call(self, inputs):
        return inputs[:, -1, :]
    
    def get_config(self):
        return super().get_config()

# -------------------------------
# 1. LOAD DATA
# -------------------------------

df = pd.read_csv("min.csv")

print(df.columns)

# Drop rows where AQI is missing
df = df[df["AQI"].notnull()]

# Fill missing values in input columns with median
input_cols = [col for col in df.columns if col != "AQI"]
df[input_cols] = df[input_cols].fillna(df[input_cols].median())

# -------------------------------
# 2. SCALE DATA
# -------------------------------

scaler = MinMaxScaler()
scaled_inputs = scaler.fit_transform(df[input_cols])
joblib.dump(scaler, SCALER_PATH)

target = df["AQI"].values.reshape(-1, 1)

# -------------------------------
# 3. PREPARE SEQUENCES
# -------------------------------

def create_sequences(X, y, seq_len):
   X_seq, y_seq = [], []
   for i in range(len(X) - seq_len):
       X_seq.append(X[i:i+seq_len])
       y_seq.append(y[i+seq_len])
   return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(scaled_inputs, target, SEQ_LEN)

# Split
X_train, X_val, y_train, y_val = train_test_split(
   X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False
)

# -------------------------------
# 4. BUILD HYBRID CNN + LSTM MODEL
# -------------------------------

# Input layer
input_layer = Input(shape=(SEQ_LEN, X_seq.shape[2]))

# CNN branch for local pattern extraction
cnn_branch = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
cnn_branch = BatchNormalization()(cnn_branch)
cnn_branch = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(cnn_branch)
cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
cnn_branch = Dropout(0.3)(cnn_branch)

cnn_branch = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(cnn_branch)
cnn_branch = BatchNormalization()(cnn_branch)
cnn_branch = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(cnn_branch)
cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
cnn_branch = Dropout(0.3)(cnn_branch)

# Global max pooling to get fixed-size output
cnn_features = GlobalMaxPooling1D()(cnn_branch)

# LSTM branch for temporal dependencies
lstm_branch = LSTM(64, return_sequences=True)(input_layer)
lstm_branch = Dropout(0.3)(lstm_branch)
lstm_branch = LSTM(32, return_sequences=True)(lstm_branch)
lstm_branch = Dropout(0.3)(lstm_branch)

# Attention mechanism for LSTM
# Use custom layer to get the last timestep
last_timestep = LastTimestepLayer()(lstm_branch)
query = Dense(32)(last_timestep)  # shape (batch, 1, features)
attention = Attention()([query, lstm_branch])
attention = Reshape((-1,))(attention)  # shape (batch, features)

# Get the last LSTM output using custom layer
lstm_last = LastOutputLayer()(lstm_branch)

# Combine CNN and LSTM features
combined = Concatenate()([cnn_features, attention, lstm_last])

# Dense layers for final prediction
combined = Dense(128, activation='relu')(combined)
combined = BatchNormalization()(combined)
combined = Dropout(0.4)(combined)

combined = Dense(64, activation='relu')(combined)
combined = BatchNormalization()(combined)
combined = Dropout(0.3)(combined)

combined = Dense(32, activation='relu')(combined)
combined = Dropout(0.2)(combined)

output = Dense(1)(combined)

model = Model(inputs=input_layer, outputs=output)

model.compile(
   loss='mse',
   optimizer='adam',
   metrics=['mae']
)

model.summary()

# -------------------------------
# 5. TRAIN
# -------------------------------

callbacks = [
   EarlyStopping(
       patience=PATIENCE,
       restore_best_weights=True
   ),
   ModelCheckpoint(
       MODEL_PATH,
       save_best_only=True
   )
]

history = model.fit(
   X_train, y_train,
   validation_data=(X_val, y_val),
   epochs=EPOCHS,
   batch_size=BATCH_SIZE,
   callbacks=callbacks,
   verbose=1
)

# -------------------------------
# 6. PLOTS
# -------------------------------

# Plot loss
plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title("MSE Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(os.path.join(PLOTS_DIR, "loss_curve.png"))

# Plot MAE
plt.figure()
plt.plot(history.history['mae'], label='train_mae')
plt.plot(history.history['val_mae'], label='val_mae')
plt.legend()
plt.title("MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.savefig(os.path.join(PLOTS_DIR, "mae_curve.png"))

print("Training complete. Hybrid CNN+LSTM model saved to:", MODEL_PATH)
print("Scaler saved to:", SCALER_PATH)
