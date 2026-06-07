import torch
from transformers import pipeline
import warnings

# Suppress HuggingFace warnings for cleaner logs
warnings.filterwarnings('ignore')

print("[System] Loading real GenAI Sentinel (distilgpt2) for local clients...")
generator = pipeline('text-generation', model='distilgpt2', device=-1)

def analyze_packet_features(anomaly_features, client_id):
    """
    Real integration with a HuggingFace LLM.
    We convert the numerical network features into a text prompt
    and ask the LLM to generate a cyber threat report.
    """
    mean_val = anomaly_features.mean().item()
    max_val = anomaly_features.max().item()
    
    # Construct a prompt for the LLM
    prompt = f"Network Anomaly on Client {client_id}. Mean deviation: {mean_val:.2f}, Max deviation: {max_val:.2f}. The cyber threat is "
    
    # Generate text (real LLM execution)
    out = generator(prompt, max_new_tokens=15, num_return_sequences=1, truncation=True, pad_token_id=50256)
    generated_text = out[0]['generated_text']
    
    # Clean up the output
    report = generated_text.replace('\n', ' ')
    return f"Client {client_id} LLM: {report}"
