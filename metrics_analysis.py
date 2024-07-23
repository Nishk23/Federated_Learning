import glob
import json

import pandas as pd
import plotly.graph_objects as go


def load_metrics(file_pattern):
    metrics_list = []
    for file_name in glob.glob(file_pattern):
        with open(file_name, 'r') as f:
            for line in f:
                metrics_list.append(json.loads(line))
    return metrics_list


def process_metrics(metrics_list):
    df = pd.DataFrame(metrics_list)
    summary = df.groupby("client_id").agg({
        "accuracy": ["mean", "std"],
        "precision": ["mean", "std"],
        "recall": ["mean", "std"],
        "f1": ["mean", "std"]
    }).reset_index()
    summary.columns = ['client_id', 'accuracy_mean', 'accuracy_std', 'precision_mean', 'precision_std', 'recall_mean',
                       'recall_std', 'f1_mean', 'f1_std']
    return summary


def plot_metrics(summary):
    fig = go.Figure()

    # Add traces for each metric
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']

    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            x=summary['client_id'],
            y=summary[f'{metric}_mean'],
            name=f'{metric.capitalize()} Mean',
            marker=dict(color=colors[i]),
            error_y=dict(
                type='data',
                array=summary[f'{metric}_std'],
                visible=True
            )
        ))

    # Update layout
    fig.update_layout(
        title='Client Performance Metrics',
        xaxis=dict(title='Client ID'),
        yaxis=dict(title='Metrics'),
        barmode='group',
        legend=dict(title='Metrics'),
        template='plotly_dark'
    )

    fig.show()


def main():
    metrics_list = load_metrics("metrics_client_*.json")
    summary = process_metrics(metrics_list)
    plot_metrics(summary)


if __name__ == "__main__":
    main()
