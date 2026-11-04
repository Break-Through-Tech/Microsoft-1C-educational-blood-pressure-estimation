
import pandas as pd

df = pd.read_csv("outputs/rf_features.csv")

print("Dataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nBlood pressure summary:")
print(df[["SBP", "DBP"]].describe())

print("\nPeak count comparison:")
print(
    df[
        [
            "ppg_peak_count",
            "abp_peak_count",
            "peak_count_difference"
        ]
    ].describe()
)

print("\nAverage heart rate:")
print(df["mean_heart_rate"].describe())

print("\nWindows per recording:")
print(df.groupby("record_id").size().describe())
