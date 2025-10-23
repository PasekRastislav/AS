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

# Generate synthetic metrics
data = {
    'timestamp': timestamps,
    'cpu_load': np.random.uniform(10, 95, n).round(2),
    'memory_usage': np.random.uniform(20, 99, n).round(2),
    'disk_io': np.random.uniform(50, 300, n).round(2),
    'network_io': np.random.uniform(10, 150, n).round(2)
}

# Create DataFrame
df_logs = pd.DataFrame(data)

# Save to CSV
csv_path = os.path.join(data_dir, 'synthetic_logs.csv')
df_logs.to_csv(csv_path, index=False)

print(f"Synthetic dataset saved to: {csv_path}")
