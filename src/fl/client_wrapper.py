import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict
from src.llm.sentinel_mock import analyze_packet_features
import time
import psutil
import os
import csv

class GenAISentinelClient(fl.client.NumPyClient):
    def __init__(self, cid, model, trainloader):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.semantic_reports = []

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        self.semantic_reports = []
        
        # Local training loop (1 epoch)
        for data, _ in self.trainloader:
            self.optimizer.zero_grad()
            reconstructed = self.model(data)
            loss = self.criterion(reconstructed, data)
            loss.backward()
            self.optimizer.step()
            
            # Sentinel LLM Hook: Adaptive threshold based on batch statistics
            mse = torch.mean((data - reconstructed)**2, dim=1)
            
            # Adaptive threshold: mean + 2*std of the batch reconstruction errors.
            # As the autoencoder improves, the threshold tightens around normal traffic,
            # making anomalies stand out more clearly over successive rounds.
            mse_mean = mse.mean()
            mse_std = mse.std()
            adaptive_threshold = mse_mean + 2.0 * mse_std
            
            anomaly_mask = mse > adaptive_threshold
            if anomaly_mask.any():
                # Pick the worst anomaly in the batch
                anomaly_indices = torch.where(anomaly_mask)[0]
                worst_idx = anomaly_indices[torch.argmax(mse[anomaly_indices])]
                raw_features = data[worst_idx]
                
                # Hardware Profiling Start
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss / (1024 * 1024) # MB
                start_time = time.time()
                
                report = analyze_packet_features(raw_features, self.cid)
                
                # Hardware Profiling End
                end_time = time.time()
                mem_after = process.memory_info().rss / (1024 * 1024) # MB
                latency_ms = (end_time - start_time) * 1000
                mem_spike = max(0, mem_after - mem_before)
                
                # Log to hardware_profile.csv
                log_file = "results/hardware_profile.csv"
                file_exists = os.path.isfile(log_file)
                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["client_id", "latency_ms", "memory_spike_mb"])
                    writer.writerow([self.cid, latency_ms, mem_spike])

                # Deduplicate within the same batch to save memory
                if report not in self.semantic_reports:
                    self.semantic_reports.append(report)

        reports_str = " | ".join(self.semantic_reports)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"reports": reports_str}

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
                
                # Adaptive threshold matching fit() logic
                mse_mean = mse.mean()
                mse_std = mse.std()
                adaptive_threshold = mse_mean + 2.0 * mse_std
                predictions = (mse > adaptive_threshold).long()
                
                # Metrics Calculation
                anomalies_mask = (targets == 1)
                normals_mask = (targets == 0)
                
                total_anomalies += anomalies_mask.sum().item()
                total_normals += normals_mask.sum().item()
                
                correct_detections += (predictions[anomalies_mask] == 1).sum().item()
                false_positives += (predictions[normals_mask] == 1).sum().item()
                
        avg_loss = total_loss / len(self.trainloader)
        detection_rate = correct_detections / max(total_anomalies, 1)
        fpr = false_positives / max(total_normals, 1)
        
        metrics = {
            "loss": avg_loss,
            "detection_rate": detection_rate,
            "false_positive_rate": fpr
        }
        return float(avg_loss), len(self.trainloader.dataset), metrics
