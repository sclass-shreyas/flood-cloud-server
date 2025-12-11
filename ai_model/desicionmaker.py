import pandas as pd
import numpy as np
import tensorflow as tf
import keras as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- CONFIGURATION ---
FILE_PATH = 'bangalore_flood_dataset_2000_2022.csv'
TIME_STEPS = 5  # Number of previous hours to look at (Sequence Length)
TEST_SIZE = 0.2 # 20% of data for testing
EPOCHS = 15     # Number of training iterations

# 1. Load Data
df = pd.read_csv(FILE_PATH)
print(f"Original Data Loaded. Shape: {df.shape}")

# Drop date/time info as sequence will handle time dependency
data = df.drop(columns=['Date', 'Time_Hour']).values

# Separate features (X) and target (Y)
X = data[:, :-1]  # All columns except the last one (Flood_State_Target)
Y = data[:, -1]   # The last column (Flood_State_Target)
print("Done Loading and Separating Data.")

# Run this after loading the data but before scaling
print("Columns in the final dataset (excluding Date/Time):")
print(list(df.drop(columns=['Date', 'Time_Hour', 'Flood_State_Target(0/1)']).columns))

# Assuming your DataFrame is still named 'df'
print("--- Full List of Columns in DataFrame ---")
print(df.columns.tolist())

# Run this code block after loading the CSV, before scaling:
feature_columns = [
    'Max_Temp(C)', 'Min_Temp(C)', 'Rainfall(mm/hr)', 'Humidity(%)', 
    'Wind_Speed(km/h)', 'Cloud_Coverage(Oktas)', 'Drain_Water_Level_Lag1(cm)'
]
missing_cols = [col for col in feature_columns if col not in df.columns]

if missing_cols:
    print(f"\n⚠️ WARNING: The following expected features are missing: {missing_cols}")
else:
    print("\n✅ All 7 input features are present. The feature shape should be 8.")


# 2. Normalization (Scaling)
# Scaling is crucial for Deep Learning models
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# 3. Create Time-Series Sequences (The core LSTM preparation)
# This function converts the 2D array [Samples, Features] 
# into the 3D array [Samples, Time Steps, Features]
def create_sequences(X, Y, time_steps):
    Xs, Ys = [], []
    for i in range(len(X) - time_steps):
        # Input sequence (look back 'time_steps' hours)
        Xs.append(X[i:(i + time_steps), :])
        # Target (the outcome immediately following the sequence)
        Ys.append(Y[i + time_steps])
    return np.array(Xs), np.array(Ys)

X_seq, Y_seq = create_sequences(X_scaled, Y, TIME_STEPS)

print(f"Sequence Input Shape (X_seq): {X_seq.shape}") 
print(f"Target Output Shape (Y_seq): {Y_seq.shape}") 
# Expected shape: (Total Samples, 5, 8) -> 8 features over 5 time steps

# 4. Train/Test Split (Maintain chronological order)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_seq, Y_seq, test_size=TEST_SIZE, shuffle=False
)

# 5. Define the LSTM Model Architecture
# Stacked LSTMs often capture more complex patterns

# Determine the input shape for the first LSTM layer
input_shape = (X_train.shape[1], X_train.shape[2]) # (Time Steps, Features)

model = Sequential([
    # First LSTM layer with 'return_sequences=True' to pass output to the next LSTM
    LSTM(units=64, return_sequences=True, input_shape=input_shape), 
    Dropout(0.3),
    
    # Second LSTM layer (return_sequences=False for output to the Dense layer)
    LSTM(units=32), 
    Dropout(0.3),
    
    # Output layer: 1 unit with sigmoid activation for binary classification (0 or 1)
    Dense(units=1, activation='sigmoid') 
])

# 6. Compile the model
# Use binary_crossentropy for your binary classification problem
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\n--- Model Architecture ---")
model.summary()

# 7. Train the model
print("\n--- Starting Model Training ---")
history = model.fit(
    X_train, Y_train,
    epochs=EPOCHS, 
    batch_size=64, # Adjust based on your memory
    validation_split=0.1, # Use 10% of training data for validation
    verbose=1
)

# 8. Evaluate the model on the Test Set
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print(f"\n--- Model Evaluation (Test Set) ---")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# 9. Save the trained model for deployment (connecting to the servo motor)
model.save('flood_prediction_lstm_model.h5')
print("\nModel saved as 'flood_prediction_lstm_model.h5'")

# CODE FOR STEP 1: Check Imbalance Metrics
from sklearn.metrics import classification_report

# 1. Generate predictions for the test set
Y_pred_raw = model.predict(X_test)

# 2. Convert probabilities to binary predictions (0 or 1)
# Use 0.5 as the threshold
Y_pred = (Y_pred_raw > 0.5).astype(int)

# 3. Print the classification report
print("\n--- Classification Report (Check for Imbalance) ---")
print(classification_report(Y_test, Y_pred, target_names=['No Flood (0)', 'Flood Risk (1)']))

import pickle
from sklearn.preprocessing import MinMaxScaler
# Assuming 'X' is your original, unscaled feature array (N, 7)

# --- 1. Fit the Scaler ---
# Note: You only need to run fit_transform once during training setup
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X) 
# X_scaled is the data you used to train the model

# --- 2. Save the Scaler Object ---
# Use pickle to serialize the scaler object, saving the calculated min/max values
SCALER_FILENAME = 'flood_risk_scaler.pkl'
with open(SCALER_FILENAME, 'wb') as file:
    pickle.dump(scaler, file)

print(f"✅ MinMaxScaler object saved as '{SCALER_FILENAME}'")

import numpy as np
import pickle
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
MODEL_PATH = 'flood_prediction_lstm_model.h5'
SCALER_PATH = 'flood_risk_scaler.pkl'
TIME_STEPS = 5  # Must match the training sequence length (5 hours)
NUM_FEATURES = 7 # Must match the training feature count (7 features)

# --- 1. Load Components ---
deployed_model = load_model(MODEL_PATH)

# Load the fitted scaler object
with open(SCALER_PATH, 'rb') as file:
    scaler = pickle.load(file)

print("✅ Model and Scaler loaded successfully for deployment.")

# Initialize the buffer to store the last 5 readings
# In a real system, this should load the last 5 recorded data points from a file/DB
REALTIME_BUFFER = np.zeros((TIME_STEPS, NUM_FEATURES)) 

# --- 2. Real-Time Data Preparation Function ---
def prepare_new_data(current_features_raw):
    """
    Takes 7 raw feature values, updates the history buffer, and prepares the 3D input.
    current_features_raw: A NumPy array or list of 7 values in the correct order.
    """
    global REALTIME_BUFFER
    
    # Update the buffer: Shift old data out, put new data in
    REALTIME_BUFFER = np.roll(REALTIME_BUFFER, shift=-1, axis=0) # Rolls data up one slot
    REALTIME_BUFFER[-1, :] = current_features_raw # Insert the latest 7 features
    
    # Scale the 5-hour sequence (5, 7) using the loaded scaler
    # Note: .transform() is used on new data, NOT fit_transform()
    X_scaled = scaler.transform(REALTIME_BUFFER)
    
    # Reshape for LSTM: (1 sample, 5 timesteps, 7 features)
    X_input = X_scaled.reshape(1, TIME_STEPS, NUM_FEATURES)
    
    return X_input

# --- 3. Example Usage ---
# Simulate reading a new hour's worth of data from your sensors/APIs:
# Order: [Max Temp, Min Temp, Rainfall, Humidity, Wind Speed, Cloud Coverage, Drain Water Level]
new_raw_data = np.array([30.0, 20.0, 5.0, 70.0, 9.0, 6.0, 75.0]) 

# Prepare the data for prediction
lstm_input = prepare_new_data(new_raw_data)

# Make a prediction
probability = deployed_model.predict(lstm_input, verbose=0)[0][0]

print(f"\nModel Input Shape (1, 5, 7): {lstm_input.shape}")
print(f"Predicted Flood Risk Probability: {probability:.4f}")

if probability >= 0.7:
    print("🚨 ACTUATION TRIGGERED: OPEN DRAIN GATE 🚨")
