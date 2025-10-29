import pandas as pd
import numpy as np
import os
import time
from multiprocessing import Pool, cpu_count
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from AS.ml_pipeline.src.utils.summarize_metrics import summarize_metrics

def process_batch(df_chunk):
    start_time = time.time()

    df_chunk.ffill(inplace=True)
    features = ['memory_usage', 'disk_io', 'network_io']
    X = df_chunk[features]
    y = df_chunk['cpu_load']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    elapsed = time.time() - start_time

    result_df = df_chunk.copy()
    result_df['prediction'] = y_pred

    return {'results': result_df, 'metrics': {'mse': mse, 'r2': r2, 'time_sec': elapsed}}

def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/synthetic_logs.csv')
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../results/distributed_batch_results')
    os.makedirs(results_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    n_cores = cpu_count()  # Number of CPU cores
    print("Using", n_cores, "cores for distributed processing.")

    # Split data into chunks for each process
    chunk_size = int(np.ceil(len(df) / n_cores))
    print("Total data size:", len(df), "rows. Each chunk size:", chunk_size, "rows.")
    df_chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size].copy() for i in range(n_cores)]
    print("Data split into", len(df_chunks), "chunks for processing.")

    with Pool(processes=n_cores) as pool:
        results = pool.map(process_batch, df_chunks)

    # Save results and aggregate metrics
    all_metrics = []
    for i, res in enumerate(results):
        res['results'].to_csv(os.path.join(results_dir, f'dist_batch_{i+1}_results.csv'), index=False)
        metrics = res['metrics']
        metrics['batch'] = i + 1
        all_metrics.append(metrics)
        print(f"Batch {i+1}: Time={metrics['time_sec']:.3f}s, MSE={metrics['mse']:.4f}, R^2={metrics['r2']:.4f}")

    metrics_df = pd.DataFrame(all_metrics)
    summary_path = os.path.join(results_dir, 'distributed_batch_metrics_summary.csv')
    metrics_df.to_csv(summary_path, index=False)
    summarize_metrics(all_metrics, os.path.join(results_dir, 'distributed_batch_metrics_overview.csv'))
    print("Distributed batch processing completed, results saved in", results_dir)

if __name__ == '__main__':
    main()
