import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

def plot_accuracy_vs_params(results_file, output_file):
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract data
    models = []
    params = []
    accuracies = []
    
    for model_name, data in results.items():
        models.append(model_name)
        params.append(data['params'] / 1e6)  # Convert to millions
        accuracies.append(data['accuracy'])
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(params, accuracies, s=100)
    
    # Add labels
    for i, model in enumerate(models):
        plt.annotate(model, (params[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    # Add reference lines for other architectures
    # These are example values - replace with actual numbers from your experiments
    reference_models = {
        'ResNet-50': {'params': 25.5, 'accuracy': 76.0},
        'ResNet-152': {'params': 60.2, 'accuracy': 78.3},
        'DenseNet-201': {'params': 20.0, 'accuracy': 77.3}
    }
    
    for model, data in reference_models.items():
        plt.scatter(data['params'], data['accuracy'], s=100, c='red')
        plt.annotate(model, (data['params'], data['accuracy']),
                    xytext=(5, 5), textcoords='offset points')
    
    # Style plot
    plt.xlabel('Parameters (Millions)')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.title('Model Accuracy vs. Parameters')
    plt.grid(True)
    
    # Save plot
    plt.savefig(output_file)
    plt.close()

if __name__ == '__main__':
    results_file = Path('results/model_results.json')
    output_file = Path('results/accuracy_vs_params.png')
    plot_accuracy_vs_params(results_file, output_file) 