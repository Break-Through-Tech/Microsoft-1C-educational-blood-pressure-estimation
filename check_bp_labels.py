
import pandas as pd

df = pd.read_csv("outputs/rf_features.csv")

print("\nSBP AND DBP SUMMARY")
print(df[["SBP", "DBP"]].describe())

print("\nWINDOWS WITH LOWEST SBP")
print(
    df.nsmallest(10, "SBP")[
        ["record_id", "window_number", "SBP", "DBP"]
    ].to_string(index=False)
)

print("\nWINDOWS WITH HIGHEST SBP")
print(
    df.nlargest(10, "SBP")[
        ["record_id", "window_number", "SBP", "DBP"]
    ].to_string(index=False)
)

print("\nWINDOWS WITH LOWEST DBP")
print(
    df.nsmallest(10, "DBP")[
        ["record_id", "window_number", "SBP", "DBP"]
    ].to_string(index=False)
)

print("\nWINDOWS WITH HIGHEST DBP")
print(
    df.nlargest(10, "DBP")[
        ["record_id", "window_number", "SBP", "DBP"]
    ].to_string(index=False)
)

print("\nCHECK FOR DBP GREATER THAN OR EQUAL TO SBP")
print(df[df["DBP"] >= df["SBP"]][
    ["record_id", "window_number", "SBP", "DBP"]
])
