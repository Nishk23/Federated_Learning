import glob
import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def load_last_round_metrics(file_pattern):
    metrics_list = []
    for file_name in glob.glob(file_pattern):
        try:
            with open(file_name, 'r') as f:
                lines = f.readlines()
                if not lines:
                    print(f"Warning: {file_name} is empty.")
                    continue
                last_line = lines[-1].strip()
                if last_line:
                    data = json.loads(last_line)
                    if 'client_id' not in data:
                        print(f"Warning: 'client_id' not found in {file_name}")
                    metrics_list.append(data)
                else:
                    print(f"Warning: Last line of {file_name} is empty.")
        except json.JSONDecodeError as e:
            print(f"Error reading JSON from {file_name}: {e}")
            print(f"Content: {lines[-1]}")
    return metrics_list


def process_metrics(metrics_list):
    if not metrics_list:
        print("No metrics data found.")
        return pd.DataFrame()
    df = pd.DataFrame(metrics_list)
    print("DataFrame structure:\n", df.head())  # Debugging print to check DataFrame structure
    if 'client_id' not in df.columns:
        print("'client_id' column not found in DataFrame.")
        return pd.DataFrame()
    summary = df.groupby("client_id").agg({
        "accuracy": ["mean", "std"],
        "precision": ["mean", "std"],
        "recall": ["mean", "std"],
        "f1": ["mean", "std"]
    }).reset_index()
    summary.columns = ['client_id', 'accuracy_mean', 'accuracy_std', 'precision_mean', 'precision_std', 'recall_mean',
                       'recall_std', 'f1_mean', 'f1_std']
    summary['client_id'] = summary['client_id'].astype(str)  # Ensure client_id is a string
    return summary


def plot_performance_improvement(federated_summary, isolated_summary):
    if federated_summary.empty or isolated_summary.empty:
        print("One of the summaries is empty. Cannot plot performance improvement.")
        return
    fig = go.Figure()

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']

    for i, metric in enumerate(metrics):
        improvement = federated_summary[f'{metric}_mean'] - isolated_summary[f'{metric}_mean']
        fig.add_trace(go.Bar(
            x=federated_summary['client_id'],
            y=improvement,
            name=f'{metric.capitalize()} Improvement',
            marker=dict(color=colors[i])
        ))

    fig.update_layout(
        title='Performance Improvement from Federated Learning',
        xaxis=dict(title='Client ID'),
        yaxis=dict(title='Improvement in Metrics'),
        barmode='group',
        legend=dict(title='Metrics'),
        template='plotly_dark'
    )

    fig.show()


def plot_performance_variability(federated_summary, isolated_summary):
    if federated_summary.empty or isolated_summary.empty:
        print("One of the summaries is empty. Cannot plot performance variability.")
        return
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    federated_metrics = federated_summary.melt(id_vars=['client_id'],
                                               value_vars=[f'{metric}_mean' for metric in metrics],
                                               var_name='metric',
                                               value_name='federated_value')
    isolated_metrics = isolated_summary.melt(id_vars=['client_id'],
                                             value_vars=[f'{metric}_mean' for metric in metrics],
                                             var_name='metric',
                                             value_name='isolated_value')

    df = federated_metrics.merge(isolated_metrics, on=['client_id', 'metric'])

    df_melted = df.melt(id_vars=['client_id', 'metric'],
                        value_vars=['federated_value', 'isolated_value'],
                        var_name='training_type',
                        value_name='value')

    fig = px.box(df_melted, x='metric', y='value', color='training_type',
                 title='Performance Variability', template='plotly_dark')
    fig.update_layout(
        xaxis=dict(title='Metric'),
        yaxis=dict(title='Value'),
        legend_title='Training Type'
    )

    fig.show()


def plot_convergence(federated_history, isolated_history):
    fig = go.Figure()

    for client_id, history in federated_history.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(history['accuracy']))),
            y=history['accuracy'],
            mode='lines+markers',
            name=f'Client {client_id} Federated Accuracy'
        ))

    for client_id, history in isolated_history.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(history['accuracy']))),
            y=history['accuracy'],
            mode='lines+markers',
            name=f'Client {client_id} Isolated Accuracy',
            line=dict(dash='dash')
        ))

    fig.update_layout(
        title='Convergence of Performance Metrics',
        xaxis=dict(title='Round/Epoch'),
        yaxis=dict(title='Accuracy'),
        template='plotly_dark'
    )

    fig.show()


def plot_contribution_vs_benefit(contributions, federated_summary):
    if federated_summary.empty:
        print("Federated summary is empty. Cannot plot contribution vs benefit.")
        return
    contributions['client_id'] = contributions['client_id'].astype(str)  # Ensure client_id is a string
    benefits = federated_summary.copy()
    benefits = benefits.merge(contributions, on='client_id')
    benefits['improvement'] = benefits['accuracy_mean'] - benefits['baseline_accuracy']

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=benefits['client_id'],
        y=benefits['improvement'],
        name='Improvement in Accuracy',
        marker=dict(color='#636EFA'),
        offsetgroup=0
    ))

    fig.add_trace(go.Bar(
        x=benefits['client_id'],
        y=benefits['contribution'],
        name='Contribution',
        marker=dict(color='#00CC96'),
        offsetgroup=1
    ))

    fig.update_layout(
        title='Client Contribution vs. Benefit',
        xaxis=dict(title='Client ID'),
        yaxis=dict(title='Improvement in Accuracy and Contribution', side='left'),
        barmode='group',  # Use 'group' for side-by-side bars
        legend=dict(title='Metrics'),
        template='plotly_dark'
    )

    fig.show()


def load_federated_history(base_path):
    federated_history = {}
    client_dirs = glob.glob(os.path.join(base_path, "client_*"))

    for client_dir in client_dirs:
        if "isolated" not in client_dir:  # Skip isolated clients
            client_id = client_dir.split('_')[-1]
            federated_history[client_id] = {"accuracy": []}

            # Find the last checkpoint directory
            checkpoint_dirs = sorted(glob.glob(os.path.join(client_dir, "checkpoint-*")), key=os.path.getmtime)
            if checkpoint_dirs:
                last_checkpoint_dir = checkpoint_dirs[-1]
                trainer_state_file = os.path.join(last_checkpoint_dir, "trainer_state.json")
                if os.path.exists(trainer_state_file):
                    with open(trainer_state_file, 'r') as f:
                        trainer_state = json.load(f)
                        if "log_history" in trainer_state:
                            for log_entry in trainer_state["log_history"]:
                                if "eval_accuracy" in log_entry:
                                    federated_history[client_id]["accuracy"].append(log_entry["eval_accuracy"])
                                else:
                                    print(f"Warning: eval_accuracy not found in log entry at step "
                                          f"{log_entry.get('step', 'unknown')} in {trainer_state_file}")
                else:
                    print(f"Warning: Trainer state file not found in {last_checkpoint_dir}")

    return federated_history


def load_isolated_history(base_path):
    isolated_history = {}
    client_dirs = glob.glob(os.path.join(base_path, "client_*_isolated"))

    for client_dir in client_dirs:
        client_id = client_dir.split('_')[-2]
        isolated_history[client_id] = {"accuracy": []}

        # Find the last checkpoint directory
        checkpoint_dirs = sorted(glob.glob(os.path.join(client_dir, "checkpoint-*")), key=os.path.getmtime)
        if checkpoint_dirs:
            last_checkpoint_dir = checkpoint_dirs[-1]
            trainer_state_file = os.path.join(last_checkpoint_dir, "trainer_state.json")
            if os.path.exists(trainer_state_file):
                with open(trainer_state_file, 'r') as f:
                    trainer_state = json.load(f)
                    if "log_history" in trainer_state:
                        for log_entry in trainer_state["log_history"]:
                            if "eval_accuracy" in log_entry:
                                isolated_history[client_id]["accuracy"].append(log_entry["eval_accuracy"])
                            else:
                                print(f"Warning: eval_accuracy not found in log entry at step "
                                      f"{log_entry.get('step', 'unknown')} in {trainer_state_file}")
            else:
                print(f"Warning: Trainer state file not found in {last_checkpoint_dir}")

    return isolated_history


def load_contributions(base_path):
    contributions = {
        'client_id': [],
        'contribution': [],  # This will be the number of steps or epochs
        'baseline_accuracy': []  # We will fill this later
    }

    client_dirs = glob.glob(os.path.join(base_path, "client_*"))

    for client_dir in client_dirs:
        if "isolated" not in client_dir:  # Skip isolated clients
            client_id = client_dir.split('_')[-1]
            trainer_state_files = sorted(glob.glob(os.path.join(client_dir, "checkpoint-*",
                                                                "trainer_state.json")))
            if trainer_state_files:
                # Read the last trainer_state.json file for steps/epochs
                with open(trainer_state_files[-1], 'r') as f:
                    trainer_state = json.load(f)
                    if "max_steps" in trainer_state:
                        contributions['client_id'].append(client_id)
                        contributions['contribution'].append(trainer_state["max_steps"])
                    else:
                        print(f"max_steps not found in {trainer_state_files[-1]}")
            else:
                print(f"Trainer state file not found in {client_dir}")

    return contributions


def load_isolated_baseline_accuracy(isolated_history):
    baseline_accuracy = {client_id: history['accuracy'][-1]
                         for client_id, history in isolated_history.items() if history['accuracy']}
    return baseline_accuracy


def check_contributions_lengths(contributions):
    lengths = {key: len(value) for key, value in contributions.items()}
    if len(set(lengths.values())) != 1:
        print("Inconsistent lengths found in contributions:")
        for key, length in lengths.items():
            print(f"Length of {key}: {length}")
        return False
    return True


def main():
    base_path = "results"

    federated_metrics = load_last_round_metrics("metrics_client_*.json")
    isolated_metrics = load_last_round_metrics("isolated_metrics_client_*.json")

    federated_summary = process_metrics(federated_metrics)
    isolated_summary = process_metrics(isolated_metrics)

    # Plot performance improvement
    plot_performance_improvement(federated_summary, isolated_summary)

    # Plot performance variability
    plot_performance_variability(federated_summary, isolated_summary)

    # Placeholder for history data (needs actual data from training logs)
    federated_history = load_federated_history(base_path)
    isolated_history = load_isolated_history(base_path)

    # Placeholder for contributions data (needs actual data)
    contributions = load_contributions(base_path)

    # Fill the baseline accuracy
    baseline_accuracy = load_isolated_baseline_accuracy(isolated_history)
    contributions['baseline_accuracy'] = [baseline_accuracy.get(str(client_id), None)
                                          for client_id in contributions['client_id']]

    # Check for consistent lengths
    if not check_contributions_lengths(contributions):
        print("Error: Contributions data has inconsistent lengths. Aborting.")
        return

    contributions = pd.DataFrame(contributions)

    # Plot contribution vs benefit
    plot_contribution_vs_benefit(contributions, federated_summary)


if __name__ == "__main__":
    main()
