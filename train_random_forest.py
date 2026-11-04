
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

df = pd.read_csv("outputs/rf_features.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Select PPG features and prediction targets
# --------------------------------------------------

ppg_features = [
    "ppg_mean",
    "ppg_std",
    "ppg_min",
    "ppg_max",
    "ppg_range",
    "ppg_peak_count",
    "mean_peak_interval",
    "std_peak_interval",
    "mean_heart_rate",
    "std_heart_rate",
    "mean_quality",
    "mean_peak_amplitude"
]

X = df[ppg_features]

# Predict both systolic and diastolic blood pressure
y = df[["SBP", "DBP"]]

# Keep record IDs for splitting the data
groups = df["record_id"]


# --------------------------------------------------
# 3. Split data by recording
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Training recordings:", groups.iloc[train_indices].nunique())
print("Testing recordings:", groups.iloc[test_indices].nunique())


# --------------------------------------------------
# 4. Create and train the Random Forest model
# --------------------------------------------------

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")


# --------------------------------------------------
# 5. Make predictions
# --------------------------------------------------

predictions = rf_model.predict(X_test)

# Convert predictions into a DataFrame
predictions_df = pd.DataFrame(
    predictions,
    columns=["Predicted_SBP", "Predicted_DBP"],
    index=y_test.index
)


# --------------------------------------------------
# 6. Evaluate model performance
# --------------------------------------------------

for i, target in enumerate(["SBP", "DBP"]):

    actual = y_test[target]
    predicted = predictions[:, i]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    within_10 = np.mean(
        np.abs(actual - predicted) <= 10
    ) * 100

    print(f"\n{target} Results")
    print("--------------------------")
    print(f"MAE: {mae:.2f} mmHg")
    print(f"RMSE: {rmse:.2f} mmHg")
    print(f"R²: {r2:.3f}")
    print(f"Within 10 mmHg: {within_10:.2f}%")


# --------------------------------------------------
# 7. Display feature importance
# --------------------------------------------------

importance_df = pd.DataFrame({
    "Feature": ppg_features,
    "Importance": rf_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print("--------------------------")
print(importance_df.to_string(index=False))


# --------------------------------------------------
# 8. Save predictions
# --------------------------------------------------

results = df.loc[
    y_test.index,
    ["record_id", "window_number", "SBP", "DBP"]
].copy()

results["Predicted_SBP"] = predictions[:, 0]
results["Predicted_DBP"] = predictions[:, 1]

results.to_csv(
    "outputs/rf_predictions.csv",
    index=False
)

print("\nPredictions saved to outputs/rf_predictions.csv")
