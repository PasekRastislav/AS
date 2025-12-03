import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from AS.ml_pipeline.src.utils.system_metrics import SystemMetricsTracker
import time
import os

def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/synthetic_logs.csv')
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../results/batch_results')
    os.makedirs(results_dir, exist_ok=True)

    model = LinearRegression()
    scaler = StandardScaler()
    batch_size = 100

    tracker = SystemMetricsTracker()

    total_records = 0
    sum_y = 0.0
    sum_y_sq = 0.0
    ss_res = 0.0
    mae_sum = 0.0

    total_start = time.perf_counter()

    for chunk_number, chunk in enumerate(pd.read_csv(data_path, chunksize=batch_size), start=1):
        chunk.ffill(inplace=True)
        features = ['memory_usage', 'disk_io', 'network_io']
        X = chunk[features]
        y = chunk['cpu_load']

        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)

        errors = y.values - y_pred
        ss_res += np.sum(errors ** 2)
        mae_sum += np.sum(np.abs(errors))
        sum_y += y.sum()
        sum_y_sq += np.sum(y.values ** 2)
        total_records += len(chunk)

        tracker.sample_memory()

        result_df = chunk.copy()
        result_df['prediction'] = y_pred
        result_df.to_csv(os.path.join(results_dir, f'batch_{chunk_number}_results.csv'), index=False)

    total_time = time.perf_counter() - total_start
    metrics = {
        'pipeline': 'batch',
        'records_processed': total_records,
        'mse': (ss_res / total_records) if total_records else None,
        'mae': (mae_sum / total_records) if total_records else None,
        'accuracy_r2': None,
    }

    if total_records:
        ss_tot = sum_y_sq - (sum_y ** 2) / total_records
        if ss_tot > 0:
            metrics['accuracy_r2'] = 1 - (ss_res / ss_tot)
        else:
            metrics['accuracy_r2'] = 0.0

    system_stats = tracker.finish()
    metrics.update({
        'wall_time_sec': total_time,
        'cpu_time_sec': system_stats['cpu_time_sec'],
        'avg_cpu_percent': system_stats['avg_cpu_percent'],
        'peak_memory_mb': system_stats['peak_rss_bytes'] / (1024 * 1024)
    })

    metrics_path = os.path.join(results_dir, 'pipeline_metrics.csv')
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print(f"Total processing time for batch pipeline: {total_time:.3f}s")


if __name__ == '__main__':
    main()
