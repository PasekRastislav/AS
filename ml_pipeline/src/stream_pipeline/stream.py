import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import time
import os

def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/synthetic_logs.csv')
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../results/stream_results')
    os.makedirs(results_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    model = LinearRegression()
    scaler = StandardScaler()

    features = ['memory_usage', 'disk_io', 'network_io']
    metrics = []

    train_data = []
    train_labels = []

    predictions = []

    for index, row in df.iterrows():
        start_time = time.time()

        train_data.append(row[features].values)
        train_labels.append(row['cpu_load'])

        X_train = np.array(train_data)
        y_train = np.array(train_labels)
        X_scaled = scaler.fit_transform(X_train)

        model.fit(X_scaled, y_train)
        y_pred = model.predict(X_scaled[-1].reshape(1, -1))[0]

        elapsed = time.time() - start_time

        predictions.append({
            'timestamp': row['timestamp'],
            'true_cpu_load': row['cpu_load'],
            'predicted_cpu_load': y_pred,
            'latency_sec': elapsed
        })

        if (index + 1) % 1000 == 0:
            mse = mean_squared_error(y_train[-100:], [p['predicted_cpu_load'] for p in predictions[-100:]])
            print(f"Event {index+1}: Avg Latency={np.mean([p['latency_sec'] for p in predictions[-100:]]):.5f}s, MSE={mse:.4f}")

    # After all events, calculate summary metrics
    preds = np.array([p['predicted_cpu_load'] for p in predictions])
    truths = np.array([p['true_cpu_load'] for p in predictions])
    latencies = np.array([p['latency_sec'] for p in predictions])

    overall_mse = mean_squared_error(truths, preds)
    overall_r2 = r2_score(truths, preds)
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)

    summary = {
        'total_events': len(predictions),
        'avg_latency_sec': avg_latency,
        'min_latency_sec': min_latency,
        'max_latency_sec': max_latency,
        'overall_mse': overall_mse,
        'overall_r2': overall_r2
    }

    print("Stream processing summary metrics:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    results_df = pd.DataFrame(predictions)
    results_df.to_csv(os.path.join(results_dir, 'stream_predictions.csv'), index=False)

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(results_dir, 'stream_summary_metrics.csv'), index=False)

    print("Stream processing complete. Results and summary saved in", results_dir)

if __name__ == '__main__':
    main()
