import pandas as pd
import numpy as np
from scipy.stats import kurtosis

input_path = "data/Full_data.csv"
output_path = "data/Final_training_data.csv"
window_size = 100 # samples amount from every window

df_raw = pd.read_csv(input_path, encoding="ISO-8859-1")
df_split = df_raw.iloc[:, 0].str.split(";", expand=True)
df_split.columns = ["time_ms", "current_ma", "state"]

df_split["current_ma"] = pd.to_numeric(df_split["current_ma"], errors="coerce")
df_split = df_split.dropna(subset=["current_ma"])

feature_rows = []

for i in range(0, len(df_split) - window_size + 1, window_size):
    window = df_split.iloc[i:i+window_size]
    current = window["current_ma"]
    label = window["state"].mode().values[0] # label majority of current sliding window

    features = {
        "mean_current": current.mean(),
        "std_current": current.std(),
        "max": current.max(),
        "min": current.min(),
        "peak_to_peak": current.max() - current.min(),
        "kurtosis": kurtosis(current),
        "label": label
    }
    feature_rows.append(features)

df_features = pd.DataFrame(feature_rows)
df_features.to_csv(output_path, index=False)

print(f" {len(df_features)} Samples extracted")
print(f" Saved to: {output_path}")