def summarize_metrics(metrics_list, save_path):
    import pandas as pd
    df = pd.DataFrame(metrics_list)

    summary = {
        'total_batches': len(df),
        'avg_time_sec': df['time_sec'].mean(),
        'avg_mse': df['mse'].mean(),
        'avg_r2': df['r2'].mean(),
        'min_mse': df['mse'].min(),
        'max_r2': df['r2'].max(),
    }

    summary_df = pd.DataFrame([summary])

    # Save summary to CSV
    summary_df.to_csv(save_path, index=False)
    print(f"Summary metrics saved to {save_path}")
    print(summary_df)