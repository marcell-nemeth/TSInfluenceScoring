"""
Advanced example demonstrating advanced features of TSInfluenceScoring.

This example shows:
1. Different selection methods (Top-K, Gumbel-Softmax, Threshold)
2. Custom loss functions
3. Counterfactual analysis
4. Attribution scoring
"""

import torch
import torch.nn as nn
from tsinfluencescoring import (
    TimestampSelector,
    CombinedLoss,
    CounterfactualGenerator,
    CounterfactualExplainer
)


def create_pattern_data(num_samples=50, seq_len=60, input_dim=3):
    """
    Create time series with specific patterns at certain timestamps.
    
    Pattern 1: Spike at timestep 15
    Pattern 2: Dip at timestep 40
    Target depends on presence of these patterns.
    """
    X = torch.randn(num_samples, seq_len, input_dim) * 0.5
    
    # Add spike at timestep 15
    X[:, 15, :] += torch.randn(num_samples, input_dim) * 2.0
    
    # Add dip at timestep 40
    X[:, 40, :] -= torch.randn(num_samples, input_dim) * 1.5
    
    # Target based on patterns
    spike_magnitude = X[:, 15, :].abs().mean(dim=1, keepdim=True)
    dip_magnitude = X[:, 40, :].abs().mean(dim=1, keepdim=True)
    y = spike_magnitude + dip_magnitude + torch.randn(num_samples, 1) * 0.1
    
    return X, y


def compare_selection_methods():
    """Compare different selection methods."""
    print("\n" + "=" * 60)
    print("Comparing Selection Methods")
    print("=" * 60)
    
    batch_size, seq_len, input_dim = 8, 60, 3
    k = 10
    
    X, _ = create_pattern_data(num_samples=batch_size, seq_len=seq_len, input_dim=input_dim)
    
    methods = {
        "Top-K": TimestampSelector(input_dim, selection_method="topk", k=k),
        "Gumbel-Softmax": TimestampSelector(input_dim, selection_method="gumbel", temperature=1.0),
        "Threshold": TimestampSelector(input_dim, selection_method="threshold", threshold=0.5)
    }
    
    print("\nSelection statistics by method:")
    print("-" * 60)
    
    for method_name, selector in methods.items():
        with torch.no_grad():
            mask, scores = selector(X, return_scores=True)
            stats = selector.compute_selection_stats(mask)
            
            print(f"\n{method_name}:")
            print(f"  Avg selected: {stats['num_selected']:.2f}")
            print(f"  Selection ratio: {stats['selection_ratio']:.2%}")
            print(f"  Mean score: {stats['mean_score']:.4f}")


def demonstrate_counterfactual_analysis():
    """Demonstrate counterfactual generation and analysis."""
    print("\n" + "=" * 60)
    print("Counterfactual Analysis")
    print("=" * 60)
    
    batch_size, seq_len, input_dim = 4, 60, 3
    output_dim = 1
    
    # Create data
    X, y = create_pattern_data(num_samples=batch_size, seq_len=seq_len, input_dim=input_dim)
    
    # Create selector
    selector = TimestampSelector(input_dim, selection_method="topk", k=10)
    
    # Create simple prediction model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(seq_len * input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim)
    )
    
    def model_wrapper(x):
        return model(x.reshape(x.shape[0], -1))
    
    # Get selection
    with torch.no_grad():
        mask, _ = selector(X, return_scores=False)
    
    print("\nGenerating counterfactuals with different methods...")
    print("-" * 60)
    
    # Test different counterfactual generation methods
    cf_methods = ["perturbation", "replacement", "removal"]
    
    for method in cf_methods:
        generator = CounterfactualGenerator(
            input_dim=input_dim,
            generation_method=method
        )
        
        with torch.no_grad():
            # Generate counterfactual
            cf = generator(X, mask, intervention_type="modify_selected")
            
            # Compute distance
            distance = generator.compute_counterfactual_distance(X, cf, mask, metric="l2")
            
            # Get predictions
            pred_original = model_wrapper(X)
            pred_cf = model_wrapper(cf)
            pred_change = (pred_cf - pred_original).abs().mean().item()
            
            print(f"\n{method.capitalize()}:")
            print(f"  Avg L2 distance: {distance.mean().item():.4f}")
            print(f"  Avg prediction change: {pred_change:.4f}")


def demonstrate_explainability():
    """Demonstrate explainability features."""
    print("\n" + "=" * 60)
    print("Explainability Analysis")
    print("=" * 60)
    
    batch_size, seq_len, input_dim = 2, 60, 3
    output_dim = 1
    
    # Create data
    X, y = create_pattern_data(num_samples=batch_size, seq_len=seq_len, input_dim=input_dim)
    
    # Create components
    selector = TimestampSelector(input_dim, selection_method="topk", k=10)
    generator = CounterfactualGenerator(input_dim, generation_method="removal")
    
    # Simple model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(seq_len * input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim)
    )
    
    def model_wrapper(x):
        return model(x.reshape(x.shape[0], -1))
    
    # Create explainer
    explainer = CounterfactualExplainer(generator, model_wrapper)
    
    # Get selection
    with torch.no_grad():
        mask, _ = selector(X, return_scores=False)
    
    print("\nExplaining predictions using counterfactuals...")
    print("-" * 60)
    
    # Generate explanations
    with torch.no_grad():
        explanations = explainer.explain_prediction(
            X[:1],  # Single sample
            mask[:1],
            intervention_types=["modify_selected", "remove_selected"]
        )
        
        print(f"\nOriginal prediction: {explanations['original_prediction'].item():.4f}")
        
        for intervention_type, details in explanations["interventions"].items():
            print(f"\n{intervention_type}:")
            print(f"  Counterfactual prediction: {details['counterfactual_prediction'].item():.4f}")
            print(f"  Prediction change: {details['prediction_change']:.4f}")
    
    print("\nComputing attribution scores...")
    print("-" * 60)
    
    # Compute attribution scores (slower, so use subset)
    with torch.no_grad():
        # Use only first 20 timesteps for speed
        X_small = X[:1, :20, :]
        mask_small = mask[:1, :20]
        
        # Recreate selector for smaller sequence
        selector_small = TimestampSelector(input_dim, selection_method="topk", k=5)
        mask_small, _ = selector_small(X_small, return_scores=False)
        
        # Simple model for smaller input
        model_small = nn.Linear(20 * input_dim, output_dim)
        
        def model_wrapper_small(x):
            return model_small(x.reshape(x.shape[0], -1))
        
        generator_small = CounterfactualGenerator(input_dim, generation_method="removal")
        explainer_small = CounterfactualExplainer(generator_small, model_wrapper_small)
        
        attribution = explainer_small.compute_attribution_scores(X_small, mask_small)
        
        print(f"\nAttribution scores shape: {attribution.shape}")
        print(f"Top 5 most influential timestamps: {attribution[0].argsort(descending=True)[:5].tolist()}")
        print(f"(Expected: around positions 15 and 40 based on data generation)")


def demonstrate_loss_functions():
    """Demonstrate different loss function configurations."""
    print("\n" + "=" * 60)
    print("Loss Function Analysis")
    print("=" * 60)
    
    batch_size, seq_len, input_dim = 8, 60, 3
    output_dim = 1
    
    # Create data
    X, y = create_pattern_data(num_samples=batch_size, seq_len=seq_len, input_dim=input_dim)
    
    # Create selector
    selector = TimestampSelector(input_dim, selection_method="topk", k=10)
    
    # Get selection
    mask, _ = selector(X, return_scores=False)
    
    # Create predictions (random for demonstration)
    predictions = torch.randn(batch_size, output_dim)
    
    print("\nComparing loss configurations...")
    print("-" * 60)
    
    # Different loss configurations
    configs = [
        {"alpha_task": 1.0, "alpha_diversity": 0.0, "use_mi": False},
        {"alpha_task": 1.0, "alpha_diversity": 0.1, "use_mi": False},
        {"alpha_task": 1.0, "alpha_diversity": 0.5, "use_mi": False},
    ]
    
    for i, config in enumerate(configs, 1):
        loss_fn = CombinedLoss(
            task_loss_fn=nn.MSELoss(),
            **config
        )
        
        total_loss, loss_dict = loss_fn(predictions, y, X, mask)
        
        print(f"\nConfiguration {i}:")
        print(f"  α_task={config['alpha_task']}, α_diversity={config['alpha_diversity']}")
        print(f"  Task loss: {loss_dict['task_loss']:.4f}")
        print(f"  Diversity loss: {loss_dict['diversity_loss']:.4f}")
        print(f"  Total loss: {loss_dict['total_loss']:.4f}")


def main():
    print("=" * 60)
    print("TSInfluenceScoring - Advanced Features")
    print("=" * 60)
    
    torch.manual_seed(42)  # For reproducibility
    
    # Run demonstrations
    compare_selection_methods()
    demonstrate_counterfactual_analysis()
    demonstrate_explainability()
    demonstrate_loss_functions()
    
    print("\n" + "=" * 60)
    print("Advanced examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
