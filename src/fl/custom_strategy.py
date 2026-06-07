import flwr as fl
from typing import List, Tuple, Dict, Optional
import json
import csv
import os

class SemanticAggregator(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_metrics = []
        self.global_reports = []
        # Ensure results directory exists
        os.makedirs("results", exist_ok=True)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ):
        aggregated_weights, _ = super().aggregate_fit(server_round, results, failures)
        
        round_reports = []
        for client, fit_res in results:
            reports_str = fit_res.metrics.get("reports", "")
            if reports_str:
                reports = reports_str.split(" | ")
                for r in reports:
                    if r and r not in round_reports:
                        round_reports.append(r)
        
        self.global_reports.append({
            "round": server_round,
            "reports": round_reports
        })
        
        # Save reports to JSON
        with open("results/semantic_reports.json", "w") as f:
            json.dump(self.global_reports, f, indent=4)

        return aggregated_weights, {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ):
        aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)
        
        # Aggregate custom metrics (Detection Rate, FPR)
        if not results:
            return aggregated_loss, {}

        total_examples = sum([res.num_examples for _, res in results])
        avg_dr = sum([res.metrics["detection_rate"] * res.num_examples for _, res in results]) / total_examples
        avg_fpr = sum([res.metrics["false_positive_rate"] * res.num_examples for _, res in results]) / total_examples
        
        round_metrics = {
            "round": server_round,
            "loss": aggregated_loss,
            "detection_rate": avg_dr,
            "false_positive_rate": avg_fpr
        }
        self.global_metrics.append(round_metrics)
        
        # Save metrics to CSV
        with open("results/metrics.csv", "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["round", "loss", "detection_rate", "false_positive_rate"])
            writer.writeheader()
            writer.writerows(self.global_metrics)

        return aggregated_loss, {"detection_rate": avg_dr, "false_positive_rate": avg_fpr}
