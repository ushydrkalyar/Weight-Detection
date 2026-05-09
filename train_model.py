import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import numpy as np

print("Loading Train and Test datasets...")
train_df = pd.read_csv('final_train_features.csv')
test_df = pd.read_csv('final_test_features.csv')

# Safety Check: Drop missing joints
train_df = train_df[(train_df['shoulder_torso_ratio'] > 0) & (train_df['hip_torso_ratio'] > 0)]
test_df = test_df[(test_df['shoulder_torso_ratio'] > 0) & (test_df['hip_torso_ratio'] > 0)]

# The Full Feature Set: Biology + Skeleton Ratios
features = ['height', 'gender', 'age', 'shoulder_torso_ratio', 'hip_torso_ratio', 'shoulder_hip_ratio']

# Prepare Training Set
X_train = train_df[features]
y_train = train_df['weight']

# Prepare Exact Testing Set
X_test = test_df[features]
y_test = test_df['weight']

print("Training the Biological & Skeleton AI model... 🧬🦴")
model = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# Final Exam
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n=== 🎯 CUSTOM DATASET REPORT CARD ===")
print(f"Average Error (MAE):  ±{mae:.2f} units")
print(f"Root Mean Sq Error:   ±{rmse:.2f} units")
print("=====================================\n")

joblib.dump(model, 'weight_predictor.pkl')
print("✅ Final Custom Model saved as 'weight_predictor.pkl'!")
