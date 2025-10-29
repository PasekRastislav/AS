import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import time
import os

def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/synthetic_logs.csv')
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../results/hybrid_results')
    os.makedirs(results_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    model = LinearRegression()
    scaler = StandardScaler()
    features = ['memory_usage', 'disk_io', 'network_io']

    batch_size = 1000  # veľkosť mikro dávky
    metrics = []
    all_predictions = []

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size].copy()
        start_time = time.time()

        batch.ffill(inplace=True)

        X = batch[features]
        y = batch['cpu_load']

        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)

        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        elapsed = time.time() - start_time

        print(f"Micro-batch {i//batch_size+1}: Time={elapsed:.3f}s, MSE={mse:.4f}, R^2={r2:.4f}")

        batch['prediction'] = y_pred
        all_predictions.append(batch)

        metrics.append({
            'micro_batch': i//batch_size + 1,
            'time_sec': elapsed,
            'mse': mse,
            'r2': r2
        })

    results_df = pd.concat(all_predictions)
    results_df.to_csv(os.path.join(results_dir, 'hybrid_predictions.csv'), index=False)

    metrics_df = pd.DataFrame(metrics)
    summary = {
        'total_micro_batches': len(metrics_df),
        'avg_time_sec': metrics_df['time_sec'].mean(),
        'avg_mse': metrics_df['mse'].mean(),
        'avg_r2': metrics_df['r2'].mean()
    }
    metrics_df.to_csv(os.path.join(results_dir, 'hybrid_metrics.csv'), index=False)

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(results_dir, 'hybrid_summary.csv'), index=False)

    print("Hybrid processing complete. Results saved in", results_dir)
    print(summary_df)

if __name__ == '__main__':
    main()
