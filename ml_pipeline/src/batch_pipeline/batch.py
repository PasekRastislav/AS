import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import time
import os

def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/synthetic_logs.csv')
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../results/batch_results')
    os.makedirs(results_dir, exist_ok=True)

    model = LinearRegression()
    scaler = StandardScaler()

    batch_size = 10000
    metrics = []

    for chunk_number, chunk in enumerate(pd.read_csv(data_path, chunksize=batch_size)):
        start_time = time.time()

        chunk.ffill(inplace=True)
        features = ['memory_usage', 'disk_io', 'network_io']
        X = chunk[features]
        y = chunk['cpu_load']

        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)

        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        elapsed_time = time.time() - start_time
        print(f"Batch {chunk_number+1}: Time={elapsed_time:.3f}s, MSE={mse:.4f}, R^2={r2:.4f}")

        result_df = chunk.copy()
        result_df['prediction'] = y_pred
        result_df.to_csv(os.path.join(results_dir, f'batch_{chunk_number+1}_results.csv'), index=False)

        metrics.append({
            'batch': chunk_number + 1,
            'time_sec': elapsed_time,
            'mse': mse,
            'r2': r2
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(results_dir, 'batch_metrics_summary.csv'), index=False)

    print("Batch processing completed with results saved in", results_dir)


if __name__ == '__main__':
    main()
