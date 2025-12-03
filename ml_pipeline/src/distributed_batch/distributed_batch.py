import pandas as pd
import numpy as np
import os
import time
from multiprocessing import Pool, cpu_count
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from AS.ml_pipeline.src.utils.system_metrics import SystemMetricsTracker

def process_batch(df_chunk):
    tracker = SystemMetricsTracker()

    df_chunk.ffill(inplace=True)
    features = ['memory_usage', 'disk_io', 'network_io']
    X = df_chunk[features]
    y = df_chunk['cpu_load']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    errors = y.values - y_pred
    ss_res = np.sum(errors ** 2)
    mae_sum = np.sum(np.abs(errors))
    sum_y = y.sum()
    sum_y_sq = np.sum(y.values ** 2)

    tracker.sample_memory()
    stats = tracker.finish()

    result_df = df_chunk.copy()
    result_df['prediction'] = y_pred

    return {
        'results': result_df,
        'stats': {
            'ss_res': ss_res,
            'mae_sum': mae_sum,
            'sum_y': sum_y,
            'sum_y_sq': sum_y_sq,
            'count': len(df_chunk),
            'cpu_time_sec': stats['cpu_time_sec'],
            'peak_memory_mb': stats['peak_rss_bytes'] / (1024 * 1024)
        }
    }

def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/synthetic_logs.csv')
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../results/distributed_batch_results')
    os.makedirs(results_dir, exist_ok=True)

    tracker = SystemMetricsTracker()

    df = pd.read_csv(data_path)
    n_cores = cpu_count()  # Number of CPU cores
    print("Using", n_cores, "cores for distributed processing.")

    # Split data into chunks for each process
    chunk_size = int(np.ceil(len(df) / n_cores))
    print("Total data size:", len(df), "rows. Each chunk size:", chunk_size, "rows.")
    df_chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size].copy() for i in range(n_cores)]
    df_chunks = [chunk for chunk in df_chunks if not chunk.empty]
    print("Data split into", len(df_chunks), "chunks for processing.")

    total_start = time.perf_counter()
    with Pool(processes=n_cores) as pool:
        results = pool.map(process_batch, df_chunks)

    total_records = 0
    ss_res_total = 0.0
    mae_sum_total = 0.0
    sum_y_total = 0.0
    sum_y_sq_total = 0.0
    worker_cpu_time = 0.0
    worker_peak_memory_mb = 0.0

    for i, res in enumerate(results, start=1):
        res['results'].to_csv(os.path.join(results_dir, f'dist_batch_{i}_results.csv'), index=False)
        stats = res['stats']
        ss_res_total += stats['ss_res']
        mae_sum_total += stats['mae_sum']
        sum_y_total += stats['sum_y']
        sum_y_sq_total += stats['sum_y_sq']
        total_records += stats['count']
        worker_cpu_time += stats['cpu_time_sec']
        worker_peak_memory_mb = max(worker_peak_memory_mb, stats['peak_memory_mb'])

    total_time = time.perf_counter() - total_start
    system_stats = tracker.finish()

    mse = (ss_res_total / total_records) if total_records else None
    mae = (mae_sum_total / total_records) if total_records else None
    r2 = None
    if total_records:
        ss_tot = sum_y_sq_total - (sum_y_total ** 2) / total_records
        if ss_tot > 0:
            r2 = 1 - (ss_res_total / ss_tot)
        else:
            r2 = 0.0

    cpu_time_sec = system_stats['cpu_time_sec'] + worker_cpu_time
    peak_memory_mb = max(system_stats['peak_rss_bytes'] / (1024 * 1024), worker_peak_memory_mb)
    avg_cpu_percent = (cpu_time_sec / total_time) * 100 if total_time > 0 else 0.0

    metrics = {
        'pipeline': 'distributed_batch',
        'records_processed': total_records,
        'mse': mse,
        'mae': mae,
        'accuracy_r2': r2,
        'wall_time_sec': total_time,
        'cpu_time_sec': cpu_time_sec,
        'avg_cpu_percent': avg_cpu_percent,
        'peak_memory_mb': peak_memory_mb
    }

    metrics_path = os.path.join(results_dir, 'pipeline_metrics.csv')
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print(f"Total processing time for distributed batch pipeline: {total_time:.3f}s")

if __name__ == '__main__':
    main()
