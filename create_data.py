import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Ensure the directory exists
data_dir = 'ml_pipeline/data'
os.makedirs(data_dir, exist_ok=True)

# Parameters for data generation
n = 1000000  # number of data points
np.random.seed(42)

# Generate timestamps spaced by seconds backwards from now
timestamps = [datetime.now() - timedelta(seconds=i) for i in range(n)]
timestamps.reverse()

# Generate synthetic metrics with a linear relationship
memory_usage = np.random.uniform(20, 99, n).round(2)
disk_io = np.random.uniform(50, 300, n).round(2)
network_io = np.random.uniform(10, 150, n).round(2)

# Define cpu_load as a linear combination of the features with noise
cpu_load = (0.5 * memory_usage + 0.3 * disk_io + 0.2 * network_io +
            np.random.normal(0, 5, n)).round(2)

# Create the data dictionary
data = {
    'timestamp': timestamps,
    'cpu_load': cpu_load,
    'memory_usage': memory_usage,
    'disk_io': disk_io,
    'network_io': network_io
}

# Create DataFrame
df_logs = pd.DataFrame(data)

# Save to CSV
csv_path = os.path.join(data_dir, 'synthetic_logs.csv')
df_logs.to_csv(csv_path, index=False)

print(f"Synthetic dataset saved to: {csv_path}")