import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import numpy as np

# 1. Load your extracted features
print("Loading data...")
df = pd.read_csv('final_training_features.csv')

# 2. Separate the Inputs (X) and the Output (y)
# X = The physical measurements
X = df[['pixel_area', 'pixel_height', 'real_height_cm']]
# y = The target we want to predict
y = df['actual_weight_kg']

# 3. Split the data into a "Training" set and a "Testing" set
# We hide 20% of the data from the model so we can test its real accuracy later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build and Train the Model (The "Brain")
print("Training the Random Forest model... 🧠")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Give the model a final exam using the 20% hidden test data
predictions = model.predict(X_test)

# Calculate how far off the model is on average
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n=== 📊 MODEL REPORT CARD ===")
print(f"Average Error (MAE):  ±{mae:.2f} kg")
print(f"Root Mean Sq Error:   ±{rmse:.2f} kg")
print("============================\n")

# 6. Save the trained model to a file so we can use it live!
joblib.dump(model, 'weight_predictor.pkl')
print("✅ Model trained and saved successfully as 'weight_predictor.pkl'!")