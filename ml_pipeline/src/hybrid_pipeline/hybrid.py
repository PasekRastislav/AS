import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from AS.ml_pipeline.src.utils.system_metrics import SystemMetricsTracker
import time
import os

def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/synthetic_logs.csv')
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../results/hybrid_results')
    os.makedirs(results_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    tracker = SystemMetricsTracker()

    model = LinearRegression()
    scaler = StandardScaler()
    features = ['memory_usage', 'disk_io', 'network_io']

    batch_size = 100  # veľkosť mikro dávky
    all_predictions = []
    total_start = time.perf_counter()

    total_records = 0
    sum_y = 0.0
    sum_y_sq = 0.0
    ss_res = 0.0
    mae_sum = 0.0

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size].copy()
        batch.ffill(inplace=True)

        X = batch[features]
        y = batch['cpu_load']

        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)

        errors = y.values - y_pred
        ss_res += np.sum(errors ** 2)
        mae_sum += np.sum(np.abs(errors))
        sum_y += y.sum()
        sum_y_sq += np.sum(y.values ** 2)
        total_records += len(batch)

        tracker.sample_memory()

        batch['prediction'] = y_pred
        all_predictions.append(batch)

    if all_predictions:
        results_df = pd.concat(all_predictions)
    else:
        results_df = pd.DataFrame(columns=df.columns)
    results_df.to_csv(os.path.join(results_dir, 'hybrid_predictions.csv'), index=False)

    total_time = time.perf_counter() - total_start
    system_stats = tracker.finish()

    mse = (ss_res / total_records) if total_records else None
    mae = (mae_sum / total_records) if total_records else None
    r2 = None
    if total_records:
        ss_tot = sum_y_sq - (sum_y ** 2) / total_records
        if ss_tot > 0:
            r2 = 1 - (ss_res / ss_tot)
        else:
            r2 = 0.0

    metrics = {
        'pipeline': 'hybrid',
        'records_processed': total_records,
        'mse': mse,
        'mae': mae,
        'accuracy_r2': r2,
        'wall_time_sec': total_time,
        'cpu_time_sec': system_stats['cpu_time_sec'],
        'avg_cpu_percent': system_stats['avg_cpu_percent'],
        'peak_memory_mb': system_stats['peak_rss_bytes'] / (1024 * 1024)
    }

    metrics_path = os.path.join(results_dir, 'pipeline_metrics.csv')
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print(f"Total processing time for hybrid pipeline: {total_time:.3f}s")
    print("Hybrid processing complete. Results saved in", results_dir)

if __name__ == '__main__':
    main()
