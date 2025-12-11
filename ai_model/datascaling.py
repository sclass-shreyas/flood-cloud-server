import pickle
from sklearn.preprocessing import MinMaxScaler

# --- Assume this is where you fitted the scaler in your training script ---
# scaler = MinMaxScaler(feature_range=(0, 1))
# X_scaled = scaler.fit_transform(X)
# --------------------------------------------------------------------------

# Save the fitted scaler object
with open('flood_risk_scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("✅ MinMaxScaler object saved as 'flood_risk_scaler.pkl'")