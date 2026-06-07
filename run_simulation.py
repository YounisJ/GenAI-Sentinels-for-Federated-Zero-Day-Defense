import flwr as fl
import logging
from src.data.dataset_generator import generate_federated_data
from src.models.autoencoder import LightweightAutoencoder
from src.fl.client_wrapper import GenAISentinelClient
from src.fl.custom_strategy import SemanticAggregator

# Setup basic logging
logging.basicConfig(level=logging.INFO)

# Hyperparameters
NUM_CLIENTS = 5
NUM_ROUNDS = 10

# Generate Data
print("[System] Generating Non-IID Synthetic Dataset for Enterprise Networks...")
client_trainloaders = generate_federated_data(num_clients=NUM_CLIENTS, samples_per_client=1000)

def client_fn(cid: str) -> fl.client.Client:
    """Creates a Flower client representing a single edge device."""
    model = LightweightAutoencoder()
    # Use the pre-generated data loader for this specific client
    trainloader = client_trainloaders[int(cid)]
    return GenAISentinelClient(cid=cid, model=model, trainloader=trainloader)

def main():
    print(f"\n[System] Starting Federated Simulation with {NUM_CLIENTS} GenAI Sentinels for {NUM_ROUNDS} Rounds...")
    
    strategy = SemanticAggregator(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )
    
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1} # Allocate 1 CPU per simulated client
    )
    
    print("\n[System] Simulation Complete. Results saved to `results/metrics.csv` and `results/semantic_reports.json`.")

if __name__ == "__main__":
    main()
