import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import StandardScaler

def generate_federated_data(num_clients=5, samples_per_client=2000, input_dim=38, anomaly_ratio=0.05):
    """
    Downloads real KDDCup99 network intrusion dataset, preprocesses it, 
    and partitions it into non-IID client dataloaders using Dirichlet allocation.
    """
    print("Downloading KDD Cup 99 Dataset (10%)...")
    kdd = fetch_kddcup99(percent10=True)
    X = kdd.data
    y = kdd.target
    
    # KDD has 41 features. 3 are categorical (protocol_type, service, flag).
    # For a lightweight autoencoder, we drop the categorical columns (indices 1, 2, 3).
    X = np.delete(X, [1, 2, 3], axis=1).astype(float)
    
    # Binary classification: b'normal.' is 0, everything else is 1 (attack)
    y_binary = np.array([0 if label == b'normal.' else 1 for label in y])
    
    # Get unique attack types for non-IID partitioning
    attack_types = np.unique(y)
    attack_type_indices = {}
    for at in attack_types:
        attack_type_indices[at] = np.where(y == at)[0]
    
    normal_indices = np.where(y_binary == 0)[0]
    anomaly_indices = np.where(y_binary == 1)[0]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # ---- Non-IID Dirichlet Partitioning ----
    # Use a low alpha (0.5) for significant heterogeneity across clients.
    # This means each client gets a different distribution of attack types.
    alpha = 0.5
    
    # Separate anomaly data by attack category for non-IID assignment
    unique_attacks = [at for at in attack_types if at != b'normal.']
    
    # Allocate anomaly proportions per client via Dirichlet
    if len(unique_attacks) > 0:
        dirichlet_proportions = np.random.dirichlet([alpha] * num_clients, size=len(unique_attacks))
    
    client_loaders = []
    
    for i in range(num_clients):
        n_normal = int(samples_per_client * (1 - anomaly_ratio))
        n_anomaly = samples_per_client - n_normal
        
        # Sample normal data (shared across clients, slight variation)
        idx_norm = np.random.choice(normal_indices, n_normal, replace=True)
        
        # Non-IID anomaly sampling: each client gets different attack type distributions
        anomaly_pool = []
        for j, at in enumerate(unique_attacks):
            at_indices = attack_type_indices[at]
            # Number of samples from this attack type for this client
            n_from_type = max(1, int(n_anomaly * dirichlet_proportions[j][i]))
            if len(at_indices) > 0:
                sampled = np.random.choice(at_indices, min(n_from_type, len(at_indices)), replace=True)
                anomaly_pool.extend(sampled)
        
        # Ensure we have exactly n_anomaly samples
        if len(anomaly_pool) >= n_anomaly:
            idx_anom = np.array(anomaly_pool[:n_anomaly])
        else:
            # Pad with random anomalies if not enough
            extra = np.random.choice(anomaly_indices, n_anomaly - len(anomaly_pool), replace=True)
            idx_anom = np.array(anomaly_pool + list(extra))
        
        X_client = np.vstack((X[idx_norm], X[idx_anom]))
        y_client = np.hstack((y_binary[idx_norm], y_binary[idx_anom]))
        
        # Shuffle
        indices = np.arange(len(y_client))
        np.random.shuffle(indices)
        X_client = X_client[indices]
        y_client = y_client[indices]
        
        X_tensor = torch.FloatTensor(X_client)
        y_tensor = torch.LongTensor(y_client)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        client_loaders.append(loader)
        
    return client_loaders
