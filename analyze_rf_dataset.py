
import pandas as pd

df = pd.read_csv("outputs/rf_features.csv")

# Show windows with the biggest difference
print("\nWINDOWS WITH LARGEST PEAK COUNT DIFFERENCES")

largest_differences = df.sort_values(
    "peak_count_difference",
    ascending=False
)

print(
    largest_differences[
        [
            "record_id",
            "window_number",
            "ppg_peak_count",
            "abp_peak_count",
            "peak_count_difference",
            "mean_heart_rate",
            "SBP",
            "DBP"
        ]
    ].head(10).to_string(index=False)
)


# Show average values for each dataset part
df["part"] = df["record_id"].str.extract(
    r"(Part_\d+)"
)

print("\nRESULTS BY DATASET PART")

grouped = df.groupby("part").agg(
    recordings=("record_id", "nunique"),
    windows=("record_id", "size"),
    avg_heart_rate=("mean_heart_rate", "mean"),
    avg_ppg_peaks=("ppg_peak_count", "mean"),
    avg_abp_peaks=("abp_peak_count", "mean"),
    avg_peak_difference=("peak_count_difference", "mean"),
    avg_sbp=("SBP", "mean"),
    avg_dbp=("DBP", "mean")
)

print(grouped.round(2))


# Check how many windows have large differences
print("\nPEAK COUNT DIFFERENCE COUNTS")

print(
    df["peak_count_difference"]
    .value_counts()
    .sort_index()
)
