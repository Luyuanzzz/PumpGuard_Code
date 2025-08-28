import numpy as np
import pandas as pd

np.random.seed(42)

# basic settings
num_samples_per_class = 200
window_size = 100
label_noise_ratio = 0.05

def generate_sample(mean, std, noise_level, label):
    signal = np.random.normal(mean, std, window_size)
    noise = np.random.normal(0, noise_level, window_size)
    current = signal + noise

    mean_val = np.mean(current)
    std_val = np.std(current)
    max_val = np.max(current)
    min_val = np.min(current)
    ptp_val = np.ptp(current)
    kurtosis = pd.Series(current).kurtosis()
    return [mean_val, std_val, max_val, min_val, ptp_val, kurtosis, label]

# Simulating 6 kinds senarios
classes = {
    "normal":         (0.45, 0.02, 0.01),
    "leakage":        (0.48, 0.03, 0.03),
    "mild_leakage":   (0.465, 0.025, 0.02),
    "clogging":       (0.30, 0.01, 0.01),
    "mild_clogging":  (0.32, 0.015, 0.015),
    "fault":          (0.10, 0.005, 0.08),
}

# generating labels
data = []
labels = list(classes.keys())

for label, (mean, std, noise) in classes.items():
    for _ in range(num_samples_per_class):
        sample = generate_sample(mean, std, noise, label)
        data.append(sample)

# Transfer to DataFrame
columns = ['mean_current', 'std_current', 'max', 'min', 'peak_to_peak', 'kurtosis', 'label']
df = pd.DataFrame(data, columns=columns)

# Adding Noises (5%)
def corrupt_labels(df, noise_ratio=0.05):
    df_noisy = df.copy()
    num_noisy = int(noise_ratio * len(df))
    indices = np.random.choice(df.index, num_noisy, replace=False)
    for idx in indices:
        current_label = df_noisy.loc[idx, 'label']
        possible_labels = [l for l in labels if l != current_label]
        df_noisy.loc[idx, 'label'] = np.random.choice(possible_labels)
    return df_noisy

df_noisy = corrupt_labels(df, label_noise_ratio)
df_noisy.to_csv("data/pump_data_generated.csv", index=False)

print("Sample Pumpdata gernerated as：pump_data_generated.csv")
