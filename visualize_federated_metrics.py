import glob
import json

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_metrics_over_rounds(file_pattern):
    metrics_dict = {}
    for file_name in glob.glob(file_pattern):
        client_id = int(file_name.split('_')[-1].split('.')[0])
        with open(file_name, 'r') as f:
            lines = f.readlines()
            metrics_dict[client_id] = [json.loads(line) for line in lines]
    return metrics_dict


def plot_metrics_over_rounds(metrics_dict):
    # Flatten the metrics into a DataFrame
    records = []
    for client_id, metrics_list in metrics_dict.items():
        for round_num, metrics in enumerate(metrics_list, start=1):
            record = {
                'client_id': client_id,
                'round': round_num,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1']
            }
            records.append(record)

    df = pd.DataFrame(records)

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy Over Rounds', 'Precision Over Rounds', 'Recall Over Rounds', 'F1 Score Over Rounds')
    )

    # Plot accuracy
    for client_id in df['client_id'].unique():
        client_df = df[df['client_id'] == client_id]
        fig.add_trace(
            go.Scatter(x=client_df['round'], y=client_df['accuracy'], mode='lines+markers',
                       name=f'Client {client_id} Accuracy'),
            row=1, col=1
        )

    # Plot precision
    for client_id in df['client_id'].unique():
        client_df = df[df['client_id'] == client_id]
        fig.add_trace(
            go.Scatter(x=client_df['round'], y=client_df['precision'], mode='lines+markers',
                       name=f'Client {client_id} Precision'),
            row=1, col=2
        )

    # Plot recall
    for client_id in df['client_id'].unique():
        client_df = df[df['client_id'] == client_id]
        fig.add_trace(
            go.Scatter(x=client_df['round'], y=client_df['recall'], mode='lines+markers',
                       name=f'Client {client_id} Recall'),
            row=2, col=1
        )

    # Plot F1 score
    for client_id in df['client_id'].unique():
        client_df = df[df['client_id'] == client_id]
        fig.add_trace(
            go.Scatter(x=client_df['round'], y=client_df['f1'], mode='lines+markers',
                       name=f'Client {client_id} F1 Score'),
            row=2, col=2
        )

    # Update layout
    fig.update_layout(
        height=800, width=1000,
        title_text='Federated Learning Metrics Over Rounds',
        template='plotly_dark'
    )

    # Update axes titles
    fig.update_xaxes(title_text='Round', row=1, col=1)
    fig.update_yaxes(title_text='Accuracy', row=1, col=1)

    fig.update_xaxes(title_text='Round', row=1, col=2)
    fig.update_yaxes(title_text='Precision', row=1, col=2)

    fig.update_xaxes(title_text='Round', row=2, col=1)
    fig.update_yaxes(title_text='Recall', row=2, col=1)

    fig.update_xaxes(title_text='Round', row=2, col=2)
    fig.update_yaxes(title_text='F1 Score', row=2, col=2)

    fig.show()


def main():
    federated_metrics = load_metrics_over_rounds("metrics_client_*.json")
    plot_metrics_over_rounds(federated_metrics)


if __name__ == "__main__":
    main()
