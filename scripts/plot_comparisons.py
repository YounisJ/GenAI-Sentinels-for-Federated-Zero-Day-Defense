import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

def plot_detection_rate():
    fedavg = pd.read_csv("results/baseline_fedavg.csv")
    fedprox = pd.read_csv("results/baseline_fedprox.csv")
    genai = pd.read_csv("results/metrics.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Detection Rate
    ax1.plot(fedavg['round'], fedavg['detection_rate'], label='FedAvg', marker='x', linestyle='--', color='#2196F3')
    ax1.plot(fedprox['round'], fedprox['detection_rate'], label='FedProx', marker='s', linestyle='--', color='#FF9800')
    ax1.plot(genai['round'], genai['detection_rate'], label='GenAI Sentinel (Ours)', marker='o', linewidth=2.5, color='#4CAF50')
    ax1.set_title('Zero-Day Detection Rate vs. Communication Rounds', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Global Communication Round')
    ax1.set_ylabel('Detection Rate')
    ax1.set_ylim([0, 0.6])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # False Positive Rate
    ax2.plot(fedavg['round'], fedavg['false_positive_rate'], label='FedAvg', marker='x', linestyle='--', color='#2196F3')
    ax2.plot(fedprox['round'], fedprox['false_positive_rate'], label='FedProx', marker='s', linestyle='--', color='#FF9800')
    ax2.plot(genai['round'], genai['false_positive_rate'], label='GenAI Sentinel (Ours)', marker='o', linewidth=2.5, color='#4CAF50')
    ax2.set_title('False Positive Rate vs. Communication Rounds', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Global Communication Round')
    ax2.set_ylabel('False Positive Rate')
    ax2.set_ylim([0, 0.03])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig("results/detection_rate_comparison.png", dpi=300, bbox_inches='tight')
    print("Generated results/detection_rate_comparison.png")
    plt.close()

def plot_loss_convergence():
    fedavg = pd.read_csv("results/baseline_fedavg.csv")
    fedprox = pd.read_csv("results/baseline_fedprox.csv")
    genai = pd.read_csv("results/metrics.csv")
    
    plt.figure(figsize=(8, 5))
    plt.plot(fedavg['round'], fedavg['loss'], label='FedAvg', marker='x', linestyle='--', color='#2196F3')
    plt.plot(fedprox['round'], fedprox['loss'], label='FedProx', marker='s', linestyle='--', color='#FF9800')
    plt.plot(genai['round'], genai['loss'], label='GenAI Sentinel (Ours)', marker='o', linewidth=2.5, color='#4CAF50')
    plt.title('Reconstruction Loss Convergence on KDD Cup 99', fontsize=12, fontweight='bold')
    plt.xlabel('Global Communication Round')
    plt.ylabel('Reconstruction Loss (MSE)')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("results/loss_curve.png", dpi=300, bbox_inches='tight')
    print("Generated results/loss_curve.png")
    plt.close()

def plot_hardware():
    hw = pd.read_csv("results/hardware_profile.csv")
    avg_latency = hw['latency_ms'].mean()
    avg_mem = hw['memory_spike_mb'].mean()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Latency histogram
    ax1.hist(hw['latency_ms'], bins=30, color='#5C6BC0', edgecolor='white', alpha=0.85)
    ax1.axvline(avg_latency, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_latency:.0f}ms')
    ax1.set_title('DistilGPT2 Inference Latency Distribution', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Latency (ms)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Memory histogram
    ax2.hist(hw['memory_spike_mb'], bins=30, color='#FF7043', edgecolor='white', alpha=0.85)
    ax2.axvline(avg_mem, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_mem:.1f}MB')
    ax2.set_title('RAM Overhead per LLM Invocation', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Memory Spike (MB)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/hardware_overhead.png", dpi=300, bbox_inches='tight')
    print("Generated results/hardware_overhead.png")
    plt.close()

if __name__ == "__main__":
    plot_detection_rate()
    plot_loss_convergence()
    plot_hardware()
