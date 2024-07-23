import flwr as fl


def main():
    # Define a strategy with metric aggregation
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample all clients for each round
        # fraction_eval=1.0,  # Evaluate all clients
        min_fit_clients=3,  # Minimum number of clients to train on
        # min_eval_clients=3,  # Minimum number of clients to evaluate on
        min_available_clients=3,  # Minimum number of clients available for training
    )

    # Start Flower server
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
