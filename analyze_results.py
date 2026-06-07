import pandas as pd
import matplotlib.pyplot as plt
import json
import os

def generate_graphs():
    if not os.path.exists('results/metrics.csv'):
        print("Error: metrics.csv not found. Run the simulation first.")
        return

    df = pd.read_csv('results/metrics.csv')

    # Plot 1: Federated Reconstruction Loss
    plt.figure(figsize=(8, 5))
    plt.plot(df['round'], df['loss'], marker='o', linestyle='-', color='b')
    plt.title('Federated Autoencoder Training Loss (Anomaly Detection)')
    plt.xlabel('Federated Round')
    plt.ylabel('Reconstruction Loss (MSE)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/loss_curve.png')
    print("Saved: results/loss_curve.png")

    # Plot 2: Zero-Day Detection Rate
    plt.figure(figsize=(8, 5))
    plt.plot(df['round'], df['detection_rate'] * 100, marker='s', linestyle='-', color='g', label='Detection Rate (TPR)')
    plt.plot(df['round'], df['false_positive_rate'] * 100, marker='x', linestyle='--', color='r', label='False Positive Rate (FPR)')
    plt.title('Zero-Day Intrusion Detection Metrics Over Time')
    plt.xlabel('Federated Round')
    plt.ylabel('Percentage (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/detection_metrics.png')
    print("Saved: results/detection_metrics.png")

def print_threat_intelligence():
    if not os.path.exists('results/semantic_reports.json'):
        return

    with open('results/semantic_reports.json', 'r') as f:
        reports = json.load(f)

    print("\n=======================================================")
    print(" GLOBAL THREAT INTELLIGENCE SUMMARY (FROM LLM SENTINELS) ")
    print("=======================================================\n")
    
    for round_data in reports:
        print(f"--- Round {round_data['round']} ---")
        if not round_data['reports']:
            print("  No anomalies detected.")
        for r in round_data['reports']:
            print(f"  {r}")
        print("")

if __name__ == "__main__":
    generate_graphs()
    print_threat_intelligence()
