import os
import subprocess
import sys

import pandas as pd

def run_script(path):
    print(f"Spúšťam skript: {path}")
    result = subprocess.run([sys.executable, path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Chyba pri spúšťaní {path}:")
        print(result.stderr)
        sys.exit(1)
    print(f"Skript {path} úspešne dokončený.\n")

def main():
    scripts = [
        ('batch', 'batch_pipeline/batch.py', '../results/batch_results/pipeline_metrics.csv'),
        ('distributed_batch', 'distributed_batch/distributed_batch.py', '../results/distributed_batch_results/pipeline_metrics.csv'),
        ('stream', 'stream_pipeline/stream.py', '../results/stream_results/pipeline_metrics.csv'),
        ('hybrid', 'hybrid_pipeline/hybrid.py', '../results/hybrid_results/pipeline_metrics.csv')
    ]

    for _, script, _ in scripts:
        run_script(script)

    metrics_frames = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.normpath(os.path.join(base_dir, '../results'))
    os.makedirs(results_dir, exist_ok=True)

    for pipeline_name, _, metrics_rel_path in scripts:
        metrics_path = os.path.normpath(os.path.join(base_dir, metrics_rel_path))
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            if 'pipeline' not in df.columns:
                df['pipeline'] = pipeline_name
            metrics_frames.append(df)
        else:
            print(f"Varovanie: nenašiel som súbor s metrikami pre {pipeline_name}: {metrics_path}")

    if metrics_frames:
        combined = pd.concat(metrics_frames, ignore_index=True)
        combined_path = os.path.join(results_dir, 'all_pipeline_metrics.csv')
        combined.to_csv(combined_path, index=False)
        print(f"Všetky metriky boli zlúčené do {combined_path}")
    else:
        print("Neboli nájdené žiadne metriky na zlúčenie.")

if __name__ == "__main__":
    main()
