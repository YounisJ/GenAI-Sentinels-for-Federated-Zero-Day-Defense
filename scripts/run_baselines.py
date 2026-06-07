import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict
import logging
import csv
import os
from src.data.dataset_generator import generate_federated_data
from src.models.autoencoder import LightweightAutoencoder

logging.basicConfig(level=logging.INFO)

NUM_CLIENTS = 5
NUM_ROUNDS = 20

class BaselineClient(fl.client.NumPyClient):
    """Standard FL client with adaptive threshold and proper metric computation."""
    def __init__(self, cid, model, trainloader):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for data, _ in self.trainloader:
            self.optimizer.zero_grad()
            reconstructed = self.model(data)
            loss = self.criterion(reconstructed, data)
            loss.backward()
            self.optimizer.step()
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0.0
        correct_detections = 0
        total_anomalies = 0
        false_positives = 0
        total_normals = 0
        
        with torch.no_grad():
            for data, targets in self.trainloader:
                reconstructed = self.model(data)
                loss = self.criterion(reconstructed, data)
                total_loss += loss.item()
                mse = torch.mean((data - reconstructed)**2, dim=1)
                
                # Adaptive threshold — same logic as GenAI client
                mse_mean = mse.mean()
                mse_std = mse.std()
                adaptive_threshold = mse_mean + 2.0 * mse_std
                predictions = (mse > adaptive_threshold).long()
                
                anomalies_mask = (targets == 1)
                normals_mask = (targets == 0)
                total_anomalies += anomalies_mask.sum().item()
                total_normals += normals_mask.sum().item()
                correct_detections += (predictions[anomalies_mask] == 1).sum().item()
                false_positives += (predictions[normals_mask] == 1).sum().item()
                
        avg_loss = total_loss / len(self.trainloader)
        detection_rate = correct_detections / max(total_anomalies, 1)
        fpr = false_positives / max(total_normals, 1)
        
        return float(avg_loss), len(self.trainloader.dataset), {
            "detection_rate": detection_rate,
            "false_positive_rate": fpr
        }


class BaselineStrategy(fl.server.strategy.FedAvg):
    """Strategy that properly aggregates detection metrics."""
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.all_metrics = []
        os.makedirs("results", exist_ok=True)

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)
        
        if not results:
            return aggregated_loss, {}

        total_examples = sum([res.num_examples for _, res in results])
        avg_dr = sum([res.metrics.get("detection_rate", 0) * res.num_examples for _, res in results]) / total_examples
        avg_fpr = sum([res.metrics.get("false_positive_rate", 0) * res.num_examples for _, res in results]) / total_examples
        
        self.all_metrics.append({
            "round": server_round,
            "loss": aggregated_loss,
            "detection_rate": avg_dr,
            "false_positive_rate": avg_fpr
        })
        
        with open(f"results/baseline_{self.name}.csv", "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["round", "loss", "detection_rate", "false_positive_rate"])
            writer.writeheader()
            writer.writerows(self.all_metrics)

        return aggregated_loss, {"detection_rate": avg_dr, "false_positive_rate": avg_fpr}


# Generate data once, shared across baselines
print("Downloading KDD Cup 99 Dataset (10%)...")
client_trainloaders = generate_federated_data(num_clients=NUM_CLIENTS, samples_per_client=2000)

def client_fn(cid: str) -> fl.client.Client:
    return BaselineClient(cid=cid, model=LightweightAutoencoder(), trainloader=client_trainloaders[int(cid)])

def run_baseline(name):
    print(f"\n[System] Starting {name} Baseline for {NUM_ROUNDS} rounds...")
    strategy = BaselineStrategy(
        name=name.lower(),
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS, min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )
    
    fl.simulation.start_simulation(
        client_fn=client_fn, num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy, client_resources={"num_cpus": 1}
    )
    print(f"[System] {name} Complete. Saved to results/baseline_{name.lower()}.csv")

if __name__ == "__main__":
    run_baseline("FedAvg")
    run_baseline("FedProx")
